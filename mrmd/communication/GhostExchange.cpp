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

#include "GhostExchange.hpp"

#include <format>

#include "assert/assert.hpp"

namespace mrmd::communication::impl
{
struct PositiveNegativeCounter
{
    idx_t positive = 0;
    idx_t negative = 0;

    KOKKOS_INLINE_FUNCTION
    PositiveNegativeCounter() = default;
    KOKKOS_INLINE_FUNCTION
    PositiveNegativeCounter(const PositiveNegativeCounter& rhs) = default;
    KOKKOS_INLINE_FUNCTION
    PositiveNegativeCounter& operator=(const PositiveNegativeCounter& rhs) = default;

    KOKKOS_INLINE_FUNCTION
    PositiveNegativeCounter& operator+=(const PositiveNegativeCounter& src)
    {
        positive += src.positive;
        negative += src.negative;
        return *this;
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
IndexView GhostExchange::createGhostAtoms(data::Atoms& atoms,
                                          const data::Subdomain& subdomain,
                                          const AXIS& axis)
{
    auto pos = atoms.getPos();

    auto h_numberOfAtomsToCommunicate =
        Kokkos::create_mirror_view(Kokkos::HostSpace(), numberOfAtomsToCommunicate_);

    idx_t newAtoms = 0;
    do
    {
        // reset selected atoms
        util::grow(atomsToCommunicateAll_, newAtoms);
        Kokkos::deep_copy(atomsToCommunicateAll_, -1);
        Kokkos::deep_copy(numberOfAtomsToCommunicate_, 0);

        auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms + atoms.numGhostAtoms);
        auto kernel =
            KOKKOS_CLASS_LAMBDA(const idx_t idx, PositiveNegativeCounter& update, const bool final)
        {
            if (pos(idx, to_underlying(axis)) < subdomain.minInnerCorner[to_underlying(axis)])
            {
                if (final && (update.positive < idx_c(atomsToCommunicateAll_.extent(0))))
                {
                    atomsToCommunicateAll_(update.positive, 0) = idx;
                }
                update.positive += 1;
            }

            if (pos(idx, to_underlying(axis)) >= subdomain.maxInnerCorner[to_underlying(axis)])
            {
                if (final && (update.negative < idx_c(atomsToCommunicateAll_.extent(0))))
                {
                    atomsToCommunicateAll_(update.negative, 1) = idx;
                }
                update.negative += 1;
            }

            if (idx == atoms.numLocalAtoms + atoms.numGhostAtoms - 1)
            {
                numberOfAtomsToCommunicate_(0) = update.positive;
                numberOfAtomsToCommunicate_(1) = update.negative;
            }
        };
        Kokkos::parallel_scan(
            std::format("GhostExchange::selectAtoms_{}", to_underlying(axis)), policy, kernel);
        Kokkos::fence();
        Kokkos::deep_copy(h_numberOfAtomsToCommunicate, numberOfAtomsToCommunicate_);
        newAtoms = std::max(h_numberOfAtomsToCommunicate(0), h_numberOfAtomsToCommunicate(1));
    } while (newAtoms > idx_c(atomsToCommunicateAll_.extent(0)));

    atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms + h_numberOfAtomsToCommunicate(0) +
                 h_numberOfAtomsToCommunicate(1));
    pos = atoms.getPos();

    util::grow(correspondingRealAtom_,
               atoms.numLocalAtoms + atoms.numGhostAtoms + h_numberOfAtomsToCommunicate(0) +
                   h_numberOfAtomsToCommunicate(1));

    {
        auto policy = Kokkos::RangePolicy<>(0, newAtoms);
        auto kernel = KOKKOS_CLASS_LAMBDA(const idx_t idx)
        {
            if (idx < numberOfAtomsToCommunicate_(0))
            {
                auto newGhostIdx = atoms.numLocalAtoms + atoms.numGhostAtoms + idx;
                atoms.copy(newGhostIdx, atomsToCommunicateAll_(idx, 0));
                pos(newGhostIdx, to_underlying(axis)) += subdomain.diameter[to_underlying(axis)];
                MRMD_DEVICE_ASSERT_GREATEREQUAL(pos(newGhostIdx, to_underlying(axis)),
                                                subdomain.maxCorner[to_underlying(axis)]);
                MRMD_DEVICE_ASSERT_LESSEQUAL(pos(newGhostIdx, to_underlying(axis)),
                                             subdomain.maxGhostCorner[to_underlying(axis)]);
                auto realIdx = atomsToCommunicateAll_(idx, 0);
                while (correspondingRealAtom_(realIdx) != -1)
                {
                    realIdx = correspondingRealAtom_(realIdx);
                    MRMD_DEVICE_ASSERT_LESSEQUAL(0, realIdx);
                    MRMD_DEVICE_ASSERT_LESS(realIdx, atoms.numLocalAtoms + atoms.numGhostAtoms);
                }
                correspondingRealAtom_(newGhostIdx) = realIdx;
            }

            if (idx < numberOfAtomsToCommunicate_(1))
            {
                auto newGhostIdx = atoms.numLocalAtoms + atoms.numGhostAtoms +
                                   numberOfAtomsToCommunicate_(0) + idx;
                atoms.copy(newGhostIdx, atomsToCommunicateAll_(idx, 1));
                pos(newGhostIdx, to_underlying(axis)) -= subdomain.diameter[to_underlying(axis)];
                MRMD_DEVICE_ASSERT_LESSEQUAL(pos(newGhostIdx, to_underlying(axis)),
                                             subdomain.minCorner[to_underlying(axis)]);
                MRMD_DEVICE_ASSERT_GREATEREQUAL(pos(newGhostIdx, to_underlying(axis)),
                                                subdomain.minGhostCorner[to_underlying(axis)]);
                auto realIdx = atomsToCommunicateAll_(idx, 1);
                while (correspondingRealAtom_(realIdx) != -1)
                {
                    realIdx = correspondingRealAtom_(realIdx);
                    MRMD_DEVICE_ASSERT_LESSEQUAL(0, realIdx);
                    MRMD_DEVICE_ASSERT_LESS(realIdx, atoms.numLocalAtoms + atoms.numGhostAtoms);
                }
                correspondingRealAtom_(newGhostIdx) = realIdx;
            }
        };
        Kokkos::parallel_for(
            std::format("GhostExchange::copyAtoms_{}", to_underlying(axis)), policy, kernel);
        Kokkos::fence();
    }
    atoms.numGhostAtoms += h_numberOfAtomsToCommunicate(0) + h_numberOfAtomsToCommunicate(1);

    return correspondingRealAtom_;
}

IndexView GhostExchange::createGhostAtomsXYZ(data::Atoms& atoms, const data::Subdomain& subdomain)
{
    resetCorrespondingRealAtoms(atoms);
    atoms.numGhostAtoms = 0;

    createGhostAtoms(atoms, subdomain, AXIS::X);
    createGhostAtoms(atoms, subdomain, AXIS::Y);
    createGhostAtoms(atoms, subdomain, AXIS::Z);

    return correspondingRealAtom_;
}

void GhostExchange::resetCorrespondingRealAtoms(data::Atoms& atoms)
{
    util::grow(correspondingRealAtom_, atoms.numLocalAtoms);
    Kokkos::deep_copy(correspondingRealAtom_, -1);
}

GhostExchange::GhostExchange(const idx_t& initialSize)
    : atomsToCommunicateAll_("atomsToCommunicateAll", initialSize),
      numberOfAtomsToCommunicate_("numberOfAtomsToCommunicate"),
      correspondingRealAtom_("correspondingRealAtom", initialSize)
{
}

}  // namespace mrmd::communication::impl
