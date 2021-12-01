#include "MultiResPeriodicGhostExchange.hpp"

#include <fmt/format.h>

#include "assert.hpp"

namespace mrmd::communication::impl
{
struct PositiveNegativeCounter
{
    idx_t positiveAtoms = 0;
    idx_t positiveMolecules = 0;
    idx_t negativeAtoms = 0;
    idx_t negativeMolecules = 0;

    KOKKOS_INLINE_FUNCTION
    PositiveNegativeCounter() = default;
    KOKKOS_INLINE_FUNCTION
    PositiveNegativeCounter(const PositiveNegativeCounter& rhs) = default;

    KOKKOS_INLINE_FUNCTION
    PositiveNegativeCounter& operator+=(const PositiveNegativeCounter& src)
    {
        negativeAtoms += src.negativeAtoms;
        negativeMolecules += src.negativeMolecules;
        positiveAtoms += src.positiveAtoms;
        positiveMolecules += src.positiveMolecules;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile PositiveNegativeCounter& src) volatile
    {
        negativeAtoms += src.negativeAtoms;
        negativeMolecules += src.negativeMolecules;
        positiveAtoms += src.positiveAtoms;
        positiveMolecules += src.positiveMolecules;
    }
};
}  // namespace mrmd::communication::impl

namespace Kokkos
{  // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<mrmd::communication::impl::PositiveNegativeCounter>
{
    KOKKOS_FORCEINLINE_FUNCTION static mrmd::communication::impl::PositiveNegativeCounter sum()
    {
        return mrmd::communication::impl::PositiveNegativeCounter();
    }
};
}  // namespace Kokkos

namespace mrmd::communication::impl
{
IndexView MultiResPeriodicGhostExchange::createGhostAtoms(data::Molecules& molecules,
                                                          data::Atoms& atoms,
                                                          const data::Subdomain& subdomain,
                                                          const idx_t& dim)
{
    ASSERT_LESSEQUAL(0, dim);
    ASSERT_LESS(dim, 3);

    auto moleculesPos = molecules.getPos();
    auto moleculesNumAtoms = molecules.getNumAtoms();

    auto h_numberOfAtomsToCommunicate =
        Kokkos::create_mirror_view(Kokkos::HostSpace(), numberOfAtomsToCommunicate_);

    idx_t newMolecules = 0;
    do
    {
        // reset selected atoms
        util::grow(atomsToCommunicateAll_, newMolecules);
        Kokkos::deep_copy(atomsToCommunicateAll_, -1);
        Kokkos::deep_copy(numberOfAtomsToCommunicate_, 0);

        auto policy =
            Kokkos::RangePolicy<>(0, molecules.numLocalMolecules + molecules.numGhostMolecules);
        auto kernel =
            KOKKOS_CLASS_LAMBDA(const idx_t idx, PositiveNegativeCounter& update, const bool final)
        {
            if (moleculesPos(idx, dim) < subdomain.minInnerCorner[dim])
            {
                if (final && (update.positiveMolecules < idx_c(atomsToCommunicateAll_.extent(0))))
                {
                    atomsToCommunicateAll_(update.positiveMolecules, 0) = idx;
                    atomsToCommunicateAll_(update.positiveMolecules, 1) = update.positiveAtoms;
                }
                update.positiveMolecules += 1;
                update.positiveAtoms += moleculesNumAtoms(idx);
            }

            if (moleculesPos(idx, dim) >= subdomain.maxInnerCorner[dim])
            {
                if (final && (update.negativeMolecules < idx_c(atomsToCommunicateAll_.extent(0))))
                {
                    atomsToCommunicateAll_(update.negativeMolecules, 2) = idx;
                    atomsToCommunicateAll_(update.negativeMolecules, 3) = update.negativeAtoms;
                }
                update.negativeMolecules += 1;
                update.negativeAtoms += moleculesNumAtoms(idx);
            }

            if (idx == molecules.numLocalMolecules + molecules.numGhostMolecules - 1)
            {
                numberOfAtomsToCommunicate_(0) = update.positiveMolecules;
                numberOfAtomsToCommunicate_(1) = update.positiveAtoms;
                numberOfAtomsToCommunicate_(2) = update.negativeMolecules;
                numberOfAtomsToCommunicate_(3) = update.negativeAtoms;
            }
        };
        Kokkos::parallel_scan(
            fmt::format("MultiResPeriodicGhostExchange::selectAtoms_{}", dim), policy, kernel);
        Kokkos::fence();
        Kokkos::deep_copy(h_numberOfAtomsToCommunicate, numberOfAtomsToCommunicate_);
        newMolecules = std::max(h_numberOfAtomsToCommunicate(POSITIVE_MOLECULES),
                                h_numberOfAtomsToCommunicate(NEGATIVE_MOLECULES));
    } while (newMolecules > idx_c(atomsToCommunicateAll_.extent(0)));

    std::cout << h_numberOfAtomsToCommunicate(POSITIVE_MOLECULES) << std::endl;
    std::cout << h_numberOfAtomsToCommunicate(NEGATIVE_MOLECULES) << std::endl;
    std::cout << h_numberOfAtomsToCommunicate(POSITIVE_ATOMS) << std::endl;
    std::cout << h_numberOfAtomsToCommunicate(NEGATIVE_ATOMS) << std::endl;

    molecules.resize(molecules.numLocalMolecules + molecules.numGhostMolecules +
                     h_numberOfAtomsToCommunicate(POSITIVE_MOLECULES) +
                     h_numberOfAtomsToCommunicate(NEGATIVE_MOLECULES));
    atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms +
                 h_numberOfAtomsToCommunicate(POSITIVE_ATOMS) +
                 h_numberOfAtomsToCommunicate(NEGATIVE_ATOMS));

    util::grow(atomsCorrespondingRealAtom_,
               atoms.numLocalAtoms + atoms.numGhostAtoms +
                   h_numberOfAtomsToCommunicate(POSITIVE_ATOMS) +
                   h_numberOfAtomsToCommunicate(NEGATIVE_ATOMS));

    {
        auto moleculesPos = molecules.getPos();
        auto moleculesAtomsOffset = molecules.getAtomsOffset();
        auto moleculesNumAtoms = molecules.getNumAtoms();

        auto atomsPos = atoms.getPos();

        auto policy = Kokkos::RangePolicy<>(0, newMolecules);
        auto kernel = KOKKOS_CLASS_LAMBDA(const idx_t& idx)
        {
            if (idx < numberOfAtomsToCommunicate_(POSITIVE_MOLECULES))
            {
                auto moleculeIdx = atomsToCommunicateAll_(idx, POSITIVE_MOLECULES);

                auto atomsStart = moleculesAtomsOffset(moleculeIdx);          /// inclusive
                auto atomsEnd = atomsStart + moleculesNumAtoms(moleculeIdx);  /// exclusive
                auto moleculeSize = atomsEnd - atomsStart;

                auto moleculeNewGhostIdx =
                    molecules.numLocalMolecules + molecules.numGhostMolecules + moleculeIdx;
                ASSERT_LESS(moleculeNewGhostIdx, molecules.size());

                auto atomNewGhostIdx = atoms.numLocalAtoms + atoms.numGhostAtoms +
                                       atomsToCommunicateAll_(idx, POSITIVE_ATOMS);

                molecules.copy(moleculeNewGhostIdx, moleculeIdx);
                moleculesPos(moleculeNewGhostIdx, dim) += subdomain.diameter[dim];
                moleculesAtomsOffset(moleculeNewGhostIdx) = atomNewGhostIdx;
                moleculesNumAtoms(moleculeNewGhostIdx) = moleculeSize;
                ASSERT_GREATEREQUAL(moleculesPos(moleculeNewGhostIdx, dim),
                                    subdomain.maxCorner[dim]);
                ASSERT_LESSEQUAL(moleculesPos(moleculeNewGhostIdx, dim),
                                 subdomain.maxGhostCorner[dim]);

                for (idx_t atomIdx = 0; atomIdx < moleculeSize; ++atomIdx)
                {
                    atoms.copy(atomNewGhostIdx + atomIdx, atomsStart + atomIdx);
                    atomsPos(atomNewGhostIdx + atomIdx, dim) += subdomain.diameter[dim];
                    auto realIdx = atomsStart + atomIdx;
                    while (atomsCorrespondingRealAtom_(realIdx) != -1)
                    {
                        realIdx = atomsCorrespondingRealAtom_(realIdx);
                        ASSERT_LESSEQUAL(0, realIdx);
                        ASSERT_LESS(realIdx, atoms.numLocalAtoms + atoms.numGhostAtoms);
                    }
                    atomsCorrespondingRealAtom_(atomNewGhostIdx + atomIdx) = realIdx;
                }
            }

            if (idx < numberOfAtomsToCommunicate_(NEGATIVE_MOLECULES))
            {
                auto moleculeIdx = atomsToCommunicateAll_(idx, NEGATIVE_MOLECULES);

                auto atomsStart = moleculesAtomsOffset(moleculeIdx);          /// inclusive
                auto atomsEnd = atomsStart + moleculesNumAtoms(moleculeIdx);  /// exclusive
                auto moleculeSize = atomsEnd - atomsStart;

                auto moleculeNewGhostIdx =
                    molecules.numLocalMolecules + molecules.numGhostMolecules +
                    numberOfAtomsToCommunicate_(POSITIVE_MOLECULES) + moleculeIdx;
                ASSERT_LESS(moleculeNewGhostIdx, molecules.size());

                auto atomNewGhostIdx = atoms.numLocalAtoms + atoms.numGhostAtoms +
                                       numberOfAtomsToCommunicate_(POSITIVE_ATOMS) +
                                       atomsToCommunicateAll_(idx, NEGATIVE_ATOMS);

                molecules.copy(moleculeNewGhostIdx, moleculeIdx);
                moleculesPos(moleculeNewGhostIdx, dim) -= subdomain.diameter[dim];
                moleculesAtomsOffset(moleculeNewGhostIdx) = atomNewGhostIdx;
                moleculesNumAtoms(moleculeNewGhostIdx) = moleculeSize;
                ASSERT_LESSEQUAL(moleculesPos(moleculeNewGhostIdx, dim), subdomain.minCorner[dim]);
                ASSERT_GREATEREQUAL(moleculesPos(moleculeNewGhostIdx, dim),
                                    subdomain.minGhostCorner[dim]);

                for (idx_t atomIdx = 0; atomIdx < moleculeSize; ++atomIdx)
                {
                    atoms.copy(atomNewGhostIdx + atomIdx, atomsStart + atomIdx);
                    atomsPos(atomNewGhostIdx + atomIdx, dim) -= subdomain.diameter[dim];
                    auto realIdx = atomsStart + atomIdx;
                    while (atomsCorrespondingRealAtom_(realIdx) != -1)
                    {
                        realIdx = atomsCorrespondingRealAtom_(realIdx);
                        ASSERT_LESSEQUAL(0, realIdx);
                        ASSERT_LESS(realIdx, atoms.numLocalAtoms + atoms.numGhostAtoms);
                    }
                    atomsCorrespondingRealAtom_(atomNewGhostIdx + atomIdx) = realIdx;
                }
            }
        };
        Kokkos::parallel_for(
            fmt::format("MultiResPeriodicGhostExchange::copyAtoms_{}", dim), policy, kernel);
        Kokkos::fence();
    }
    molecules.numGhostMolecules += h_numberOfAtomsToCommunicate(POSITIVE_MOLECULES) +
                                   h_numberOfAtomsToCommunicate(NEGATIVE_MOLECULES);
    atoms.numGhostAtoms +=
        h_numberOfAtomsToCommunicate(POSITIVE_ATOMS) + h_numberOfAtomsToCommunicate(NEGATIVE_ATOMS);

    return atomsCorrespondingRealAtom_;
}

IndexView MultiResPeriodicGhostExchange::createGhostAtomsXYZ(data::Molecules& molecules,
                                                             data::Atoms& atoms,
                                                             const data::Subdomain& subdomain)
{
    resetCorrespondingRealAtoms(atoms);
    atoms.numGhostAtoms = 0;

    createGhostAtoms(molecules, atoms, subdomain, COORD_X);
    createGhostAtoms(molecules, atoms, subdomain, COORD_Y);
    createGhostAtoms(molecules, atoms, subdomain, COORD_Z);

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
    : atomsToCommunicateAll_("atomsToCommunicateAll", initialSize),
      numberOfAtomsToCommunicate_("numberOfAtomsToCommunicate"),
      atomsCorrespondingRealAtom_("atomsCorrespondingRealAtom", 0),
      moleculesCorrespondingRealMolecule_("moleculesCorrespondingRealMolecule", 0)
{
}

}  // namespace mrmd::communication::impl
