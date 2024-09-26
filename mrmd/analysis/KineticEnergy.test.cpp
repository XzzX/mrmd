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

#include "KineticEnergy.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"

namespace mrmd
{
TEST(KineticEnergy, Simple)
{
    data::Atoms atoms(3);
    auto d_AoSoA = atoms.getAoSoA();
    auto h_AoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), d_AoSoA);

    auto vel = Cabana::slice<data::Atoms::VEL>(h_AoSoA);
    auto mass = Cabana::slice<data::Atoms::MASS>(h_AoSoA);

    vel(0, 0) = +2_r;
    vel(0, 1) = +0_r;
    vel(0, 2) = +0_r;
    mass(0) = 1_r;
    vel(1, 0) = -0_r;
    vel(1, 1) = -8_r;
    vel(1, 2) = -0_r;
    mass(1) = 2_r;
    vel(2, 0) = +0_r;
    vel(2, 1) = +0_r;
    vel(2, 2) = +16_r;
    mass(2) = 0.5_r;

    Cabana::deep_copy(d_AoSoA, h_AoSoA);

    atoms.numLocalAtoms = 3;
    atoms.numGhostAtoms = 0;

    auto energy = analysis::getKineticEnergy(atoms);

    EXPECT_FLOAT_EQ(energy, (4_r + 2_r * 64_r + 0.5_r * 256_r) * 0.5_r);
}
}  // namespace mrmd