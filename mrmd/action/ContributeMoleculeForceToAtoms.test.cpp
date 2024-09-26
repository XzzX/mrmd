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

#include "ContributeMoleculeForceToAtoms.hpp"

#include <gtest/gtest.h>

#include "test/DiamondFixture.hpp"

namespace mrmd
{
namespace action
{
using ContributeMoleculeForceToAtomsTest = test::DiamondFixture;

TEST_F(ContributeMoleculeForceToAtomsTest, update)
{
    {
        auto hAoSoA =
            Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), molecules.getAoSoA());
        auto force = Cabana::slice<data::Molecules::FORCE>(hAoSoA);
        force(0, 0) = 5_r;
        force(0, 1) = 6_r;
        force(0, 2) = 7_r;
        Cabana::deep_copy(molecules.getAoSoA(), hAoSoA);
    }

    action::ContributeMoleculeForceToAtoms::update(molecules, atoms);

    {
        auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
        auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);
        EXPECT_FLOAT_EQ(force(0, 0), 5_r * 0.25_r);
        EXPECT_FLOAT_EQ(force(0, 1), 6_r * 0.25_r);
        EXPECT_FLOAT_EQ(force(0, 2), 7_r * 0.25_r);
        EXPECT_FLOAT_EQ(force(1, 0), 5_r * 0.75_r);
        EXPECT_FLOAT_EQ(force(1, 1), 6_r * 0.75_r);
        EXPECT_FLOAT_EQ(force(1, 2), 7_r * 0.75_r);
    }
}

}  // namespace action
}  // namespace mrmd