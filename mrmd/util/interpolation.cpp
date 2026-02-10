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
void updateInterpolate(const data::MultiHistogram& target, const data::MultiHistogram& input)
{
    MRMD_HOST_ASSERT_EQUAL(target.numHistograms, input.numHistograms);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({idx_t(0), idx_t(0)},
                                                         {target.numBins, input.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        // find the two enclosing bins in the input histogram
        real_t outputBinPosition = target.getBinPosition(binIdx);
        idx_t leftBinIdx = input.getBin(outputBinPosition - 0.5_r * input.binSize);
        idx_t rightBinIdx = leftBinIdx + 1;

        // only update if within bounds of input histogram or exactly at boundary
        if (leftBinIdx >= 0 && rightBinIdx < input.numBins)
        {
            auto inputLeft = input.data(leftBinIdx, histogramIdx);
            auto inputRight = input.data(rightBinIdx, histogramIdx);

            target.data(binIdx, histogramIdx) +=
                lerp(inputLeft,
                     inputRight,
                     (outputBinPosition - input.getBinPosition(leftBinIdx)) * input.inverseBinSize);
        }
        else if (isFloatEQ(outputBinPosition, input.getBinPosition(0)))
        {
            target.data(binIdx, histogramIdx) += input.data(0, histogramIdx);
        }
        else if (isFloatEQ(outputBinPosition, input.getBinPosition(input.numBins - 1)))
        {
            target.data(binIdx, histogramIdx) += input.data(input.numBins - 1, histogramIdx);
        }
    };
    Kokkos::parallel_for("MultiHistogram::updateInterpolate", policy, kernel);
    Kokkos::fence();
}

}  // namespace util
}  // namespace mrmd