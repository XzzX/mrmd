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

#include "util/interpolation.hpp"

#include <gtest/gtest.h>

#include "util/IsInSymmetricSlab.hpp"

namespace mrmd
{
namespace util
{
TEST(interpolate, preZeroInner)
{
    data::MultiHistogram histogramInput("histogramInput", 0_r, 10_r, 10, 3);
    data::MultiHistogram histogramTarget("histogramTarget", 0.5_r, 9.5_r, 30, 3);
    data::MultiHistogram histogramRef("histogramRef", histogramTarget);

    auto h_dataInput = Kokkos::create_mirror_view(histogramInput.data);
    for (auto idx = 0; idx < 10; ++idx)
    {
        h_dataInput(idx, 0) = 0_r;
        h_dataInput(idx, 1) = 1_r;
        h_dataInput(idx, 2) = histogramInput.getBinPosition(idx);
    }
    Kokkos::deep_copy(histogramInput.data, h_dataInput);

    auto h_dataTarget = Kokkos::create_mirror_view(histogramTarget.data);
    for (auto idx = 0; idx < 30; ++idx)
    {
        h_dataTarget(idx, 0) = 0_r;
        h_dataTarget(idx, 1) = 0_r;
        h_dataTarget(idx, 2) = 0_r;
    }
    Kokkos::deep_copy(histogramTarget.data, h_dataTarget);

    auto h_dataRef = Kokkos::create_mirror_view(histogramRef.data);
    for (auto idx = 0; idx < 30; ++idx)
    {
        h_dataRef(idx, 0) = 0_r;
        h_dataRef(idx, 1) = 1_r;
        h_dataRef(idx, 2) = histogramRef.getBinPosition(idx);
    }
    Kokkos::deep_copy(histogramRef.data, h_dataRef);

    util::updateInterpolate(histogramTarget, histogramInput);
    h_dataTarget = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), histogramTarget.data);

    for (auto idx = 0; idx < 30; ++idx)
    {
        EXPECT_FLOAT_EQ(h_dataTarget(idx, 0), h_dataRef(idx, 0));
        EXPECT_FLOAT_EQ(h_dataTarget(idx, 1), h_dataRef(idx, 1));
        EXPECT_FLOAT_EQ(h_dataTarget(idx, 2), h_dataRef(idx, 2));
    }
}

TEST(interpolate, nonZeroWithBoundary)
{
    data::MultiHistogram histogramInput("histogramInput", 0_r, 10_r, 10, 3);
    data::MultiHistogram histogramTarget("histogramTarget", 0_r, 10_r, 30, 3);
    data::MultiHistogram histogramRef("histogramRef", histogramTarget);

    auto h_dataInput = Kokkos::create_mirror_view(histogramInput.data);
    for (auto idx = 0; idx < 10; ++idx)
    {
        h_dataInput(idx, 0) = 0_r;
        h_dataInput(idx, 1) = 1_r;
        h_dataInput(idx, 2) = histogramInput.getBinPosition(idx);
    }
    Kokkos::deep_copy(histogramInput.data, h_dataInput);

    auto h_dataTarget = Kokkos::create_mirror_view(histogramTarget.data);
    for (auto idx = 0; idx < 30; ++idx)
    {
        h_dataTarget(idx, 0) = 1_r;
        h_dataTarget(idx, 1) = 1_r;
        h_dataTarget(idx, 2) = 1_r;
    }
    Kokkos::deep_copy(histogramTarget.data, h_dataTarget);

    auto h_dataRef = Kokkos::create_mirror_view(histogramRef.data);
    for (auto idx = 0; idx < 30; ++idx)
    {
        if (histogramRef.getBinPosition(idx) >= histogramInput.getBinPosition(0) &&
            histogramRef.getBinPosition(idx) <= histogramInput.getBinPosition(9))
        {
            h_dataRef(idx, 0) = 1_r + 0_r;
            h_dataRef(idx, 1) = 1_r + 1_r;
            h_dataRef(idx, 2) = 1_r + histogramRef.getBinPosition(idx);
        }
        else
        {
            h_dataRef(idx, 0) = 1_r + 0_r;
            h_dataRef(idx, 1) = 1_r + 0_r;
            h_dataRef(idx, 2) = 1_r + 0_r;
        }
    }
    Kokkos::deep_copy(histogramRef.data, h_dataRef);

    util::updateInterpolate(histogramTarget, histogramInput);
    h_dataTarget = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), histogramTarget.data);

    for (auto idx = 0; idx < 30; ++idx)
    {
        EXPECT_FLOAT_EQ(h_dataTarget(idx, 0), h_dataRef(idx, 0));
        EXPECT_FLOAT_EQ(h_dataTarget(idx, 1), h_dataRef(idx, 1));
        EXPECT_FLOAT_EQ(h_dataTarget(idx, 2), h_dataRef(idx, 2));
    }
}

}  // namespace util
}  // namespace mrmd