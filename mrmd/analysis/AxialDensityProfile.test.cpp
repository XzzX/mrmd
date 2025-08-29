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

#include "AxialDensityProfile.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
/**
 * increasing density of type 0 atoms from left to right
 * decreasing density of type 1 atoms from right to left
 */
data::Atoms initAtoms()
{
    data::Atoms atoms(100 * 2);
    atoms.numLocalAtoms = 200;

    auto policy = Kokkos::RangePolicy<>(0, 1);
    auto kernel = KOKKOS_LAMBDA(const idx_t& /*tmp*/, idx_t& sum)
    {
        idx_t idx = 0;
        for (auto i = 0; i < 10; ++i)
        {
            for (auto j = 0; j < i + 1; ++j)
            {
                atoms.getPos()(idx, 0) = real_c(i) + 0.5_r;
                atoms.getType()(idx) = 0;
                ++idx;

                atoms.getPos()(idx, 0) = 10_r - (real_c(i) + 0.5_r);
                atoms.getType()(idx) = 1;
                ++idx;
            }
        }
        sum += idx;
    };
    idx_t numAtoms = 0;
    Kokkos::parallel_reduce("LinearDensityProfile::histogram", policy, kernel, numAtoms);
    Kokkos::fence();
    atoms.numLocalAtoms = numAtoms;

    return atoms;
}
TEST(AxialDensityProfile, histogram)
{
    auto atoms = initAtoms();

    auto histogram = getAxialDensityProfile(
        atoms.numLocalAtoms, atoms.getPos(), atoms.getType(), 2, 0_r, 10_r, 10, 1_r, COORD_X);
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), histogram.data);

    for (auto i = 0; i < 10; ++i)
    {
        EXPECT_FLOAT_EQ(h_data(i, 0), real_c(i + 1));
        EXPECT_FLOAT_EQ(h_data(i, 1), real_c(11 - (i + 1)));
    }
}

TEST(AxialDensityProfile, overlappingBins)
{
    auto atoms = initAtoms();

    auto histogram = getAxialDensityProfile(
        atoms.numLocalAtoms, atoms.getPos(), atoms.getType(), 2, 0_r, 10_r, 10, 3_r, COORD_X);
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), histogram.data);

    for (auto i = 0; i < 10; ++i)
    {
        if (i == 0)
        {
            EXPECT_FLOAT_EQ(h_data(i, 0), (real_c(i) + real_c(i + 1) + real_c(i + 2)));
            EXPECT_FLOAT_EQ(h_data(i, 1), (real_c(11 - (i + 1)) + real_c(11 - (i + 2))));
            continue;
        }
        else if (i == 9)
        {
            EXPECT_FLOAT_EQ(h_data(i, 0), (real_c(i) + real_c(i + 1)));
            EXPECT_FLOAT_EQ(h_data(i, 1), (real_c(11 - i) + real_c(11 - (i + 1))));
            continue;
        }
        EXPECT_FLOAT_EQ(h_data(i, 0), (real_c(i) + real_c(i + 1) + real_c(i + 2)));
        EXPECT_FLOAT_EQ(h_data(i, 1),
                        (real_c(11 - i) + real_c(11 - (i + 1)) + real_c(11 - (i + 2))));
    }
}

}  // namespace analysis
}  // namespace mrmd