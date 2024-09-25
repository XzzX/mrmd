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

#include <cassert>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"

namespace mrmd
{
namespace test
{
namespace impl
{
inline void setup(data::Atoms& atoms)
{
    assert(atoms.size() >= 1);
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    auto charge = atoms.getCharge();

    auto policy = Kokkos::RangePolicy<>(0, 1);
    auto kernel = KOKKOS_LAMBDA(idx_t idx)
    {
        pos(0, 0) = +2_r;
        pos(0, 1) = +3_r;
        pos(0, 2) = +4_r;

        vel(0, 0) = +7_r;
        vel(0, 1) = +5_r;
        vel(0, 2) = +3_r;

        force(0, 0) = +9_r;
        force(0, 1) = +7_r;
        force(0, 2) = +8_r;

        mass(0) = 1.5_r;
        charge(0) = 0.5_r;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    atoms.numLocalAtoms = 1;
    atoms.numGhostAtoms = 0;
}
}  // namespace impl
class SingleAtom : public ::testing::Test
{
protected:
    void SetUp() override { impl::setup(atoms); }

    // void TearDown() override {}

    data::Atoms atoms = data::Atoms(1);
};

}  // namespace test
}  // namespace mrmd