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

#include "BerendsenBarostat.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using BerendsenBarostatTest = test::SingleAtom;

TEST_F(BerendsenBarostatTest, gamma_0)
{
    constexpr real_t gamma = 0_r;
    constexpr real_t currentPressure = 1_r;
    constexpr real_t targetPressure = 3.8_r;
    data::Subdomain subdomain({0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, 0.1_r);
    action::BerendsenBarostat::apply(atoms, currentPressure, targetPressure, gamma, subdomain);

    EXPECT_FLOAT_EQ(subdomain.maxCorner[0], 1_r);
    EXPECT_FLOAT_EQ(subdomain.maxCorner[1], 1_r);
    EXPECT_FLOAT_EQ(subdomain.maxCorner[2], 1_r);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);

    EXPECT_FLOAT_EQ(pos(0, 0), 2_r);
    EXPECT_FLOAT_EQ(pos(0, 1), 3_r);
    EXPECT_FLOAT_EQ(pos(0, 2), 4_r);
}

TEST_F(BerendsenBarostatTest, gamma_1)
{
    constexpr real_t gamma = 1_r;
    constexpr real_t currentPressure = 2_r;
    constexpr real_t targetPressure = 1_r;
    data::Subdomain subdomain({0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, 0.1_r);
    action::BerendsenBarostat::apply(atoms, currentPressure, targetPressure, gamma, subdomain);

    EXPECT_FLOAT_EQ(subdomain.maxCorner[0], std::cbrt(2_r));
    EXPECT_FLOAT_EQ(subdomain.maxCorner[1], std::cbrt(2_r));
    EXPECT_FLOAT_EQ(subdomain.maxCorner[2], std::cbrt(2_r));

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);

    EXPECT_FLOAT_EQ(pos(0, 0), 2_r * std::cbrt(2_r));
    EXPECT_FLOAT_EQ(pos(0, 1), 3_r * std::cbrt(2_r));
    EXPECT_FLOAT_EQ(pos(0, 2), 4_r * std::cbrt(2_r));
}
}  // namespace action
}  // namespace mrmd