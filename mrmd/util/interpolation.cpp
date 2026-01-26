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
data::MultiHistogram interpolate(const data::MultiHistogram& inputCoarse,
                                 const data::MultiHistogram& inputFine)
{
    MRMD_HOST_ASSERT_EQUAL(inputFine.numHistograms, inputCoarse.numHistograms);

    data::MultiHistogram output("interpolated-profile", inputFine);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {output.numBins, inputCoarse.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        // find the two enclosing bins in the input histogram
        real_t outputBinPosition = output.getBinPosition(binIdx);
        idx_t leftBinIdx = inputCoarse.getBin(outputBinPosition - 0.5_r * inputCoarse.binSize);
        idx_t rightBinIdx = leftBinIdx + 1;

        // handle boundaries
        if (leftBinIdx < 0 || rightBinIdx >= inputCoarse.numBins)
        {
            output.data(binIdx, histogramIdx) = 0_r;  // out of bounds, set to zero
            return;
        }

        auto inputDataLeft = inputCoarse.data(leftBinIdx, histogramIdx);
        auto inputDataRight = inputCoarse.data(rightBinIdx, histogramIdx);

        output.data(binIdx, histogramIdx) =
            lerp(inputDataLeft,
                 inputDataRight,
                 (outputBinPosition - inputCoarse.getBinPosition(leftBinIdx)) *
                     inputCoarse.inverseBinSize);
    };
    Kokkos::parallel_for("MultiHistogram::interpolate", policy, kernel);
    Kokkos::fence();

    return output;
}

}  // namespace util
}  // namespace mrmd