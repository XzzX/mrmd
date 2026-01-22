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

namespace mrmd
{
namespace util
{
data::MultiHistogram interpolate(const data::MultiHistogram& input, const ScalarView& grid)
{
    real_t gridSpacing = grid(1) - grid(0);
    real_t gridMin = grid(0) - 0.5_r * gridSpacing;
    real_t gridMax = grid(grid.extent(0) - 1) + 0.5_r * gridSpacing;

    data::MultiHistogram output(
        "interpolated-profile", gridMin, gridMax, idx_c(grid.extent(0)), input.numHistograms);

    MRMD_HOST_ASSERT_EQUAL(output.numBins, idx_c(grid.extent(0)), "Output grid size mismatch!");
    for (idx_t idx = 0; idx < idx_c(output.numBins); ++idx)
    {
        MRMD_HOST_ASSERT_EQUAL(output.getBinPosition(idx), grid(idx), "Output grid mismatch!");
    }

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {idx_c(grid.extent(0)), input.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        // find the two enclosing bins in the input histogram
        real_t outputBinPosition = grid(binIdx);
        idx_t leftBinIdx = input.getBin(outputBinPosition - 0.5_r * input.binSize);
        idx_t rightBinIdx = leftBinIdx + 1;

        // handle boundaries
        if (leftBinIdx < 0 || rightBinIdx >= input.numBins)
        {
            output.data(binIdx, histogramIdx) = 0_r;  // out of bounds, set to zero
            return;
        }

        auto inputDataLeft = input.data(leftBinIdx, histogramIdx);
        auto inputDataRight = input.data(rightBinIdx, histogramIdx);

        output.data(binIdx, histogramIdx) =
            lerp(inputDataLeft,
                 inputDataRight,
                 (outputBinPosition - input.getBinPosition(leftBinIdx)) * input.inverseBinSize);
    };
    Kokkos::parallel_for("MultiHistogram::interpolate", policy, kernel);
    Kokkos::fence();

    return output;
}

}  // namespace util
}  // namespace mrmd