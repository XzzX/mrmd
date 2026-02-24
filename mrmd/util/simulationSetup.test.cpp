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

#include "util/simulationSetup.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(simulationSetup, testFillDomain)
{
    data::Subdomain subdomain({1, 2, 3}, {3, 6, 9}, 0.5_r);
    data::Atoms atoms = fillDomainWithAtoms(subdomain, 100, 1_r, 1_r);

    EXPECT_EQ(atoms.size(), 100);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto mass = Cabana::slice<data::Atoms::MASS>(hAoSoA);
    auto type = Cabana::slice<data::Atoms::TYPE>(hAoSoA);
    auto charge = Cabana::slice<data::Atoms::CHARGE>(hAoSoA);

    for (idx_t idx = 0; idx < atoms.size(); ++idx)
    {
        EXPECT_GE(pos(idx, 0), subdomain.minCorner[0]);
        EXPECT_LE(pos(idx, 0), subdomain.maxCorner[0]);
        EXPECT_GE(pos(idx, 1), subdomain.minCorner[1]);
        EXPECT_LE(pos(idx, 1), subdomain.maxCorner[1]);
        EXPECT_GE(pos(idx, 2), subdomain.minCorner[2]);
        EXPECT_LE(pos(idx, 2), subdomain.maxCorner[2]);

        EXPECT_GE(vel(idx, 0), -0.5_r);
        EXPECT_LE(vel(idx, 0), +0.5_r);
        EXPECT_GE(vel(idx, 1), -0.5_r);
        EXPECT_LE(vel(idx, 1), +0.5_r);
        EXPECT_GE(vel(idx, 2), -0.5_r);
        EXPECT_LE(vel(idx, 2), +0.5_r);

        EXPECT_FLOAT_EQ(mass(idx), 1_r);
        EXPECT_EQ(type(idx), 0);
        EXPECT_FLOAT_EQ(charge(idx), 0_r);
    }
}

}  // namespace util
}  // namespace mrmd