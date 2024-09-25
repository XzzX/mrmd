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

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"

namespace mrmd
{
namespace test
{
namespace impl
{
inline void setupMolecules(data::Molecules& molecules)
{
    assert(molecules.size() >= 2);

    auto moleculesAtomsOffset = molecules.getAtomsOffset();
    auto moleculesNumAtoms = molecules.getNumAtoms();

    auto policy = Kokkos::RangePolicy<>(0, 2);
    auto kernel = KOKKOS_LAMBDA(idx_t idx)
    {
        if (idx == 0)
        {
            moleculesAtomsOffset(0) = 0;
            moleculesNumAtoms(0) = 2;
        }
        if (idx == 1)
        {
            moleculesAtomsOffset(1) = 2;
            moleculesNumAtoms(1) = 2;
        }
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    molecules.numLocalMolecules = 2;
    molecules.numGhostMolecules = 0;
}

inline void setupAtoms(data::Atoms& atoms)
{
    assert(atoms.size() >= 4);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    auto relativeMass = atoms.getRelativeMass();

    auto policy = Kokkos::RangePolicy<>(0, 4);
    auto kernel = KOKKOS_LAMBDA(idx_t idx)
    {
        if (idx == 0)
        {
            pos(0, 0) = 1_r;
            pos(0, 1) = 0_r;
            pos(0, 2) = 0_r;
            mass(0) = 1_r;
            relativeMass(0) = 0.25_r;
        }

        if (idx == 1)
        {
            pos(1, 0) = 0_r;
            pos(1, 1) = 1_r;
            pos(1, 2) = 0_r;
            mass(1) = 3_r;
            relativeMass(1) = 0.75_r;
        }
        if (idx == 2)
        {
            pos(2, 0) = -1_r;
            pos(2, 1) = 0_r;
            pos(2, 2) = 0_r;
            mass(2) = 1_r;
            relativeMass(2) = 0.25_r;
        }

        if (idx == 3)
        {
            pos(3, 0) = 0_r;
            pos(3, 1) = -1_r;
            pos(3, 2) = 0_r;
            mass(3) = 3_r;
            relativeMass(3) = 0.75_r;
        }
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    atoms.numLocalAtoms = 4;
    atoms.numGhostAtoms = 0;
}
}  // namespace impl

/**
 * 2 molecules with 2 atoms each
 * +++++
 * ++A++
 * +B+A+
 * ++B++
 * +++++
 */
class DiamondFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        impl::setupMolecules(molecules);
        impl::setupAtoms(atoms);
    }

    // void TearDown() override {}

    data::Molecules molecules = data::Molecules(2);
    data::Atoms atoms = data::Atoms(4);
};

}  // namespace test
}  // namespace mrmd