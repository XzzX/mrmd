// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "MultiResPeriodicGhostExchange.hpp"

#include <format>

#include "assert/assert.hpp"

namespace mrmd::communication::impl
{
struct MoleculesAtomsCounter
{
    idx_t positiveAtoms = 0;
    idx_t positiveMolecules = 0;
    idx_t negativeAtoms = 0;
    idx_t negativeMolecules = 0;

    KOKKOS_INLINE_FUNCTION
    MoleculesAtomsCounter() = default;
    KOKKOS_INLINE_FUNCTION
    MoleculesAtomsCounter(const MoleculesAtomsCounter& rhs) = default;
    KOKKOS_INLINE_FUNCTION
    MoleculesAtomsCounter& operator=(const MoleculesAtomsCounter& rhs) = default;

    KOKKOS_INLINE_FUNCTION
    MoleculesAtomsCounter& operator+=(const MoleculesAtomsCounter& src)
    {
        negativeAtoms += src.negativeAtoms;
        negativeMolecules += src.negativeMolecules;
        positiveAtoms += src.positiveAtoms;
        positiveMolecules += src.positiveMolecules;
        return *this;
    }
};
}  // namespace mrmd::communication::impl

namespace Kokkos
{  // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<mrmd::communication::impl::MoleculesAtomsCounter>
{
    KOKKOS_FORCEINLINE_FUNCTION static mrmd::communication::impl::MoleculesAtomsCounter sum()
    {
        return mrmd::communication::impl::MoleculesAtomsCounter();
    }
};
}  // namespace Kokkos

namespace mrmd::communication::impl
{
IndexView MultiResPeriodicGhostExchange::createGhostAtoms(data::Molecules& molecules,
                                                          data::Atoms& atoms,
                                                          const data::Subdomain& subdomain,
                                                          const AXIS& axis)
{
    auto h_numberOfCommunicationItems =
        Kokkos::create_mirror_view(Kokkos::HostSpace(), numberOfCommunicationItems_);

    idx_t newMolecules = 0;
    do
    {
        // reset selected atoms
        util::grow(communicationInfo_, newMolecules);
        Kokkos::deep_copy(communicationInfo_, -1);
        Kokkos::deep_copy(numberOfCommunicationItems_, 0);

        auto moleculesPos = molecules.getPos();
        auto moleculesNumAtoms = molecules.getNumAtoms();

        auto policy =
            Kokkos::RangePolicy<>(0, molecules.numLocalMolecules + molecules.numGhostMolecules);
        auto kernel =
            KOKKOS_CLASS_LAMBDA(const idx_t idx, MoleculesAtomsCounter& update, const bool final)
        {
            if (moleculesPos(idx, to_underlying(axis)) <
                subdomain.minInnerCorner[to_underlying(axis)])
            {
                if (final && (update.negativeMolecules < idx_c(communicationInfo_.extent(0))))
                {
                    communicationInfo_(update.negativeMolecules, Info::NEGATIVE_MOLECULE_IDX) = idx;
                    communicationInfo_(update.negativeMolecules, Info::NEGATIVE_NUM_ATOMS) =
                        update.negativeAtoms;
                }
                update.negativeMolecules += 1;
                update.negativeAtoms += moleculesNumAtoms(idx);
            }

            if (moleculesPos(idx, to_underlying(axis)) >=
                subdomain.maxInnerCorner[to_underlying(axis)])
            {
                if (final && (update.positiveMolecules < idx_c(communicationInfo_.extent(0))))
                {
                    communicationInfo_(update.positiveMolecules, Info::POSITIVE_MOLECULE_IDX) = idx;
                    communicationInfo_(update.positiveMolecules, Info::POSITIVE_NUM_ATOMS) =
                        update.positiveAtoms;
                }
                update.positiveMolecules += 1;
                update.positiveAtoms += moleculesNumAtoms(idx);
            }

            if (final && (idx == molecules.numLocalMolecules + molecules.numGhostMolecules - 1))
            {
                numberOfCommunicationItems_(Item::POSITIVE_MOLECULES) = update.positiveMolecules;
                numberOfCommunicationItems_(Item::POSITIVE_ATOMS) = update.positiveAtoms;
                numberOfCommunicationItems_(Item::NEGATIVE_MOLECULES) = update.negativeMolecules;
                numberOfCommunicationItems_(Item::NEGATIVE_ATOMS) = update.negativeAtoms;
            }
        };
        Kokkos::parallel_scan(
            std::format("MultiResPeriodicGhostExchange::selectAtoms_{}", to_underlying(axis)),
            policy,
            kernel);
        Kokkos::fence();
        Kokkos::deep_copy(h_numberOfCommunicationItems, numberOfCommunicationItems_);
        newMolecules = std::max(h_numberOfCommunicationItems(Item::POSITIVE_MOLECULES),
                                h_numberOfCommunicationItems(Item::NEGATIVE_MOLECULES));
    } while (newMolecules > idx_c(communicationInfo_.extent(0)));

    molecules.resize(molecules.numLocalMolecules + molecules.numGhostMolecules +
                     h_numberOfCommunicationItems(Item::POSITIVE_MOLECULES) +
                     h_numberOfCommunicationItems(Item::NEGATIVE_MOLECULES));
    atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms +
                 h_numberOfCommunicationItems(Item::POSITIVE_ATOMS) +
                 h_numberOfCommunicationItems(Item::NEGATIVE_ATOMS));

    util::grow(atomsCorrespondingRealAtom_,
               atoms.numLocalAtoms + atoms.numGhostAtoms +
                   h_numberOfCommunicationItems(Item::POSITIVE_ATOMS) +
                   h_numberOfCommunicationItems(Item::NEGATIVE_ATOMS));

    {
        auto moleculesPos = molecules.getPos();
        auto moleculesAtomsOffset = molecules.getAtomsOffset();
        auto moleculesNumAtoms = molecules.getNumAtoms();

        auto atomsPos = atoms.getPos();

        auto policy = Kokkos::RangePolicy<>(0, newMolecules);
        auto kernel = KOKKOS_CLASS_LAMBDA(const idx_t& idx)
        {
            if (idx < numberOfCommunicationItems_(Item::POSITIVE_MOLECULES))
            {
                auto moleculeIdx = communicationInfo_(idx, Info::POSITIVE_MOLECULE_IDX);

                auto atomsStart = moleculesAtomsOffset(moleculeIdx);          /// inclusive
                auto atomsEnd = atomsStart + moleculesNumAtoms(moleculeIdx);  /// exclusive
                auto moleculeSize = atomsEnd - atomsStart;

                auto moleculeNewGhostIdx =
                    molecules.numLocalMolecules + molecules.numGhostMolecules + idx;
                MRMD_DEVICE_ASSERT_LESS(moleculeNewGhostIdx, molecules.size());

                auto atomNewGhostIdx = atoms.numLocalAtoms + atoms.numGhostAtoms +
                                       communicationInfo_(idx, Info::POSITIVE_NUM_ATOMS);

                molecules.copy(moleculeNewGhostIdx, moleculeIdx);
                moleculesPos(moleculeNewGhostIdx, to_underlying(axis)) -=
                    subdomain.diameter[to_underlying(axis)];
                moleculesAtomsOffset(moleculeNewGhostIdx) = atomNewGhostIdx;
                moleculesNumAtoms(moleculeNewGhostIdx) = moleculeSize;
                MRMD_DEVICE_ASSERT_LESSEQUAL(moleculesPos(moleculeNewGhostIdx, to_underlying(axis)),
                                             subdomain.minCorner[to_underlying(axis)]);
                MRMD_DEVICE_ASSERT_GREATEREQUAL(
                    moleculesPos(moleculeNewGhostIdx, to_underlying(axis)),
                    subdomain.minGhostCorner[to_underlying(axis)]);

                for (idx_t atomIdx = 0; atomIdx < moleculeSize; ++atomIdx)
                {
                    atoms.copy(atomNewGhostIdx + atomIdx, atomsStart + atomIdx);
                    atomsPos(atomNewGhostIdx + atomIdx, to_underlying(axis)) -=
                        subdomain.diameter[to_underlying(axis)];
                    auto realIdx = atomsStart + atomIdx;
                    while (atomsCorrespondingRealAtom_(realIdx) != -1)
                    {
                        realIdx = atomsCorrespondingRealAtom_(realIdx);
                        MRMD_DEVICE_ASSERT_LESSEQUAL(0, realIdx);
                        MRMD_DEVICE_ASSERT_LESS(realIdx, atoms.numLocalAtoms + atoms.numGhostAtoms);
                    }
                    atomsCorrespondingRealAtom_(atomNewGhostIdx + atomIdx) = realIdx;
                }
            }

            if (idx < numberOfCommunicationItems_(Item::NEGATIVE_MOLECULES))
            {
                auto moleculeIdx = communicationInfo_(idx, Info::NEGATIVE_MOLECULE_IDX);

                auto atomsStart = moleculesAtomsOffset(moleculeIdx);          /// inclusive
                auto atomsEnd = atomsStart + moleculesNumAtoms(moleculeIdx);  /// exclusive
                auto moleculeSize = atomsEnd - atomsStart;

                auto moleculeNewGhostIdx =
                    molecules.numLocalMolecules + molecules.numGhostMolecules +
                    numberOfCommunicationItems_(Item::POSITIVE_MOLECULES) + idx;
                MRMD_DEVICE_ASSERT_LESS(moleculeNewGhostIdx, molecules.size());

                auto atomNewGhostIdx = atoms.numLocalAtoms + atoms.numGhostAtoms +
                                       numberOfCommunicationItems_(Item::POSITIVE_ATOMS) +
                                       communicationInfo_(idx, Info::NEGATIVE_NUM_ATOMS);

                molecules.copy(moleculeNewGhostIdx, moleculeIdx);
                moleculesPos(moleculeNewGhostIdx, to_underlying(axis)) +=
                    subdomain.diameter[to_underlying(axis)];
                moleculesAtomsOffset(moleculeNewGhostIdx) = atomNewGhostIdx;
                moleculesNumAtoms(moleculeNewGhostIdx) = moleculeSize;
                MRMD_DEVICE_ASSERT_GREATEREQUAL(
                    moleculesPos(moleculeNewGhostIdx, to_underlying(axis)),
                    subdomain.maxCorner[to_underlying(axis)]);
                MRMD_DEVICE_ASSERT_LESSEQUAL(moleculesPos(moleculeNewGhostIdx, to_underlying(axis)),
                                             subdomain.maxGhostCorner[to_underlying(axis)]);

                for (idx_t atomIdx = 0; atomIdx < moleculeSize; ++atomIdx)
                {
                    atoms.copy(atomNewGhostIdx + atomIdx, atomsStart + atomIdx);
                    atomsPos(atomNewGhostIdx + atomIdx, to_underlying(axis)) +=
                        subdomain.diameter[to_underlying(axis)];
                    auto realIdx = atomsStart + atomIdx;
                    while (atomsCorrespondingRealAtom_(realIdx) != -1)
                    {
                        realIdx = atomsCorrespondingRealAtom_(realIdx);
                        MRMD_DEVICE_ASSERT_LESSEQUAL(0, realIdx);
                        MRMD_DEVICE_ASSERT_LESS(realIdx, atoms.numLocalAtoms + atoms.numGhostAtoms);
                    }
                    atomsCorrespondingRealAtom_(atomNewGhostIdx + atomIdx) = realIdx;
                }
            }
        };
        Kokkos::parallel_for(
            std::format("MultiResPeriodicGhostExchange::copyAtoms_{}", to_underlying(axis)),
            policy,
            kernel);
        Kokkos::fence();
    }
    molecules.numGhostMolecules += h_numberOfCommunicationItems(Item::POSITIVE_MOLECULES) +
                                   h_numberOfCommunicationItems(Item::NEGATIVE_MOLECULES);
    atoms.numGhostAtoms += h_numberOfCommunicationItems(Item::POSITIVE_ATOMS) +
                           h_numberOfCommunicationItems(Item::NEGATIVE_ATOMS);

    return atomsCorrespondingRealAtom_;
}

IndexView MultiResPeriodicGhostExchange::createGhostAtomsXYZ(data::Molecules& molecules,
                                                             data::Atoms& atoms,
                                                             const data::Subdomain& subdomain)
{
    resetCorrespondingRealMolecules(molecules);
    molecules.numGhostMolecules = 0;

    resetCorrespondingRealAtoms(atoms);
    atoms.numGhostAtoms = 0;

    createGhostAtoms(molecules, atoms, subdomain, AXIS::X);
    createGhostAtoms(molecules, atoms, subdomain, AXIS::Y);
    createGhostAtoms(molecules, atoms, subdomain, AXIS::Z);

    return atomsCorrespondingRealAtom_;
}

void MultiResPeriodicGhostExchange::resetCorrespondingRealAtoms(data::Atoms& atoms)
{
    util::grow(atomsCorrespondingRealAtom_, atoms.numLocalAtoms);
    Kokkos::deep_copy(atomsCorrespondingRealAtom_, -1);
}

void MultiResPeriodicGhostExchange::resetCorrespondingRealMolecules(data::Molecules& molecules)
{
    util::grow(moleculesCorrespondingRealMolecule_, molecules.numLocalMolecules);
    Kokkos::deep_copy(moleculesCorrespondingRealMolecule_, -1);
}

MultiResPeriodicGhostExchange::MultiResPeriodicGhostExchange(const idx_t& initialSize)
    : communicationInfo_("atomsToCommunicateAll", initialSize),
      numberOfCommunicationItems_("numberOfAtomsToCommunicate"),
      atomsCorrespondingRealAtom_("atomsCorrespondingRealAtom", 0),
      moleculesCorrespondingRealMolecule_("moleculesCorrespondingRealMolecule", 0)
{
}

}  // namespace mrmd::communication::impl
