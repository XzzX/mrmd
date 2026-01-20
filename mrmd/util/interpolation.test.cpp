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

namespace mrmd
{
namespace util
{
TEST(interpolate, testInterpolate)
{
    data::MultiHistogram histogramCoarse("histogram", 0_r, 10_r, 10, 3);
    data::MultiHistogram histogramFine("histogram", 0.5_r, 9.5_r, 30, 3);

    auto h_dataCoarse = Kokkos::create_mirror_view(histogramCoarse.data);
    for (auto idx = 0; idx < 10; ++idx)
    {
        h_dataCoarse(idx, 0) = 0_r;
        h_dataCoarse(idx, 1) = 1_r;
        h_dataCoarse(idx, 2) = histogramCoarse.getBinPosition(idx);
    }
    Kokkos::deep_copy(histogramCoarse.data, h_dataCoarse);

    auto h_dataFine = Kokkos::create_mirror_view(histogramFine.data);
    for (auto idx = 0; idx < 30; ++idx)
    {
        h_dataFine(idx, 0) = 0_r;
        h_dataFine(idx, 1) = 1_r;
        h_dataFine(idx, 2) = histogramFine.getBinPosition(idx);
    }
    Kokkos::deep_copy(histogramFine.data, h_dataFine);

    auto histogramInterp = util::interpolate(histogramCoarse, createGrid(histogramFine));
    auto h_dataInterp = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), histogramInterp.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_dataFine(idx, 0), h_dataInterp(idx, 0));
        EXPECT_FLOAT_EQ(h_dataFine(idx, 1), h_dataInterp(idx, 1));
        EXPECT_FLOAT_EQ(h_dataFine(idx, 2), h_dataInterp(idx, 2));
    }
}
}  // namespace util
}  // namespace mrmd