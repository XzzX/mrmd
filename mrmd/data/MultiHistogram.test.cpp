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

#include "MultiHistogram.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace data
{

TEST(MultiHistogram, getBin)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 10, 2);
    EXPECT_EQ(histogram.getBin(-0.5_r), -1);
    EXPECT_EQ(histogram.getBin(0.5_r), 0);
    EXPECT_EQ(histogram.getBin(5.5_r), 5);
    EXPECT_EQ(histogram.getBin(10.5_r), -1);
}

TEST(MultiHistogram, getBinPosition)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 10, 2);
    EXPECT_FLOAT_EQ(histogram.getBinPosition(0), 0.5_r);
    EXPECT_FLOAT_EQ(histogram.getBinPosition(5), 5.5_r);
}

TEST(MultiHistogram, scale)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    h_data(5, 0) = 10_r;
    h_data(5, 1) = 5_r;
    Kokkos::deep_copy(histogram.data, h_data);

    histogram.scale(3_r);

    Kokkos::deep_copy(h_data, histogram.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_data(idx, 0), idx == 5 ? 30_r : 0_r);
        EXPECT_FLOAT_EQ(h_data(idx, 1), idx == 5 ? 15_r : 0_r);
    }
}

TEST(MultiHistogram, make_symmetric)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 10, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    for (auto idx = 0; idx < 10; ++idx)
    {
        h_data(idx, 0) = real_c(idx);
        h_data(idx, 1) = 10_r - real_c(idx);
    }

    Kokkos::deep_copy(histogram.data, h_data);

    histogram.makeSymmetric();

    Kokkos::deep_copy(h_data, histogram.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_data(idx, 0), 4.5_r);
        EXPECT_FLOAT_EQ(h_data(idx, 1), 5.5_r);
    }
}

TEST(MultiHistogram, op_plusequal)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    h_data(5, 0) = 10_r;
    h_data(5, 1) = 5_r;
    Kokkos::deep_copy(histogram.data, h_data);

    histogram += histogram;

    Kokkos::deep_copy(h_data, histogram.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_data(idx, 0), idx == 5 ? 20_r : 0_r);
        EXPECT_FLOAT_EQ(h_data(idx, 1), idx == 5 ? 10_r : 0_r);
    }
}

TEST(MultiHistogram, op_divequal)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    for (auto idx = 0; idx < 10; ++idx)
    {
        h_data(idx, 0) = -real_c(idx) - 1_r;
        h_data(idx, 1) = real_c(idx) + 1_r;
    }
    Kokkos::deep_copy(histogram.data, h_data);

    histogram /= histogram;

    Kokkos::deep_copy(h_data, histogram.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_data(idx, 0), 1_r);
        EXPECT_FLOAT_EQ(h_data(idx, 1), 1_r);
    }
}

TEST(MultiHistogram, smoothen_symmetric)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    h_data(5, 0) = 10_r;
    h_data(5, 1) = 5_r;
    Kokkos::deep_copy(histogram.data, h_data);

    auto smoothedDensityProfile = smoothen(histogram, 1_r, 3_r);

    auto h_smoothed =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), smoothedDensityProfile.data);
    for (auto idx = 1; idx < 6; ++idx)
    {
        EXPECT_FLOAT_EQ(h_smoothed(5 - idx, 0), h_smoothed(5 + idx, 0));
        EXPECT_FLOAT_EQ(h_smoothed(5 - idx, 1), h_smoothed(5 + idx, 1));
    }
}

void multiHistogramSmoothen_constant()
{
    constexpr auto CONST_VAL = 2_r;
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 1);
    Kokkos::parallel_for(
        "init_histogram", Kokkos::RangePolicy<>(0, 11), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = CONST_VAL;
        });

    auto smoothedDensityProfile = smoothen(histogram, 1_r, 3_r);

    auto h_smoothed =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), smoothedDensityProfile.data);
    for (auto idx = 0; idx < 11; ++idx)
    {
        EXPECT_FLOAT_EQ(h_smoothed(idx, 0), CONST_VAL);
    }
}
TEST(MultiHistogram, smoothen_constant) { multiHistogramSmoothen_constant(); }
}  // namespace data
}  // namespace mrmd