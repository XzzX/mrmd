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

    const ScalarView& outputGrid = createGrid(output);

    MRMD_HOST_ASSERT_EQUAL(outputGrid.extent(0), grid.extent(0), "Output grid size mismatch!");
    for (idx_t idx = 0; idx < idx_c(outputGrid.extent(0)); ++idx)
    {
        MRMD_HOST_ASSERT_EQUAL(outputGrid(idx), grid(idx), "Output grid mismatch!");
    }

    const ScalarView& inputGrid = createGrid(input);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {idx_c(grid.extent(0)), input.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        // find the two enclosing bins in the input histogram
        real_t outputBinPosition = output.getBinPosition(binIdx);
        idx_t inputBinIdx = input.getBin(outputBinPosition);
        idx_t leftBinIdx =
            inputBinIdx + idx_c(std::floor(outputBinPosition - input.getBinPosition(inputBinIdx)));
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
                 (outputBinPosition - inputGrid(leftBinIdx)) * input.inverseBinSize);
    };
    Kokkos::parallel_for("MultiHistogram::interpolate", policy, kernel);
    Kokkos::fence();

    return output;
}

data::MultiHistogram constrainToSymmetricSlab(const data::MultiHistogram& input,
                                              const util::IsInSymmetricSlab& applicationRegion)
{
    data::MultiHistogram constrainedProfile(
        "constrained-profile", input.min, input.max, input.numBins, input.numHistograms);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({idx_t(0), idx_t(0)},
                                                         {input.numBins, input.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        if (!applicationRegion(input.getBinPosition(binIdx), 0_r, 0_r))
        {
            constrainedProfile.data(binIdx, histogramIdx) = 0_r;
        }
        else
        {
            constrainedProfile.data(binIdx, histogramIdx) = input.data(binIdx, histogramIdx);
        }
    };
    Kokkos::parallel_for("MultiHistogram::constrainToSymmetricSlab", policy, kernel);
    Kokkos::fence();

    return constrainedProfile;
}
}  // namespace util
}  // namespace mrmd