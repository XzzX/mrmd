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

#include "MultiResRealAtomsExchange.hpp"

#include "assert/assert.hpp"

namespace mrmd
{
namespace communication
{
void realAtomsExchange(const data::Subdomain& subdomain,
                       const data::Molecules& molecules,
                       const data::Atoms& atoms)
{
    auto moleculesPos = molecules.getPos();
    auto atomsOffset = molecules.getAtomsOffset();
    auto numAtoms = molecules.getNumAtoms();

    auto atomsPos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
    auto kernel = KOKKOS_LAMBDA(const idx_t& moleculeIdx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            auto& moleculeX = moleculesPos(moleculeIdx, dim);
            if (subdomain.maxCorner[dim] <= moleculeX)
            {
                moleculeX -= subdomain.diameter[dim];

                auto atomsStart = atomsOffset(moleculeIdx);
                auto atomsEnd = atomsStart + numAtoms(moleculeIdx);
                for (idx_t atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
                {
                    auto& atomX = atomsPos(atomIdx, dim);
                    atomX -= subdomain.diameter[dim];
                }
                MRMD_DEVICE_ASSERT_LESS(moleculeX, subdomain.maxCorner[dim]);
                MRMD_DEVICE_ASSERT_LESSEQUAL(subdomain.minCorner[dim], moleculeX);
            }
            if (moleculeX < subdomain.minCorner[dim])
            {
                moleculeX += subdomain.diameter[dim];

                auto atomsStart = atomsOffset(moleculeIdx);
                auto atomsEnd = atomsStart + numAtoms(moleculeIdx);
                for (idx_t atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
                {
                    auto& atomX = atomsPos(atomIdx, dim);
                    atomX += subdomain.diameter[dim];
                }
                MRMD_DEVICE_ASSERT_LESS(moleculeX, subdomain.maxCorner[dim]);
                MRMD_DEVICE_ASSERT_LESSEQUAL(subdomain.minCorner[dim], moleculeX);
            }
            MRMD_DEVICE_ASSERT_LESS(moleculeX, subdomain.maxCorner[dim]);
            MRMD_DEVICE_ASSERT_LESSEQUAL(subdomain.minCorner[dim], moleculeX);
        }
    };
    Kokkos::parallel_for("realAtomsExchange::periodicMapping", policy, kernel);
    Kokkos::fence();
}

}  // namespace communication
}  // namespace mrmd