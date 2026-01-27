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
void updateInterpolate(const data::MultiHistogram& inputData, data::MultiHistogram& inputTarget)
{
    MRMD_HOST_ASSERT_EQUAL(inputTarget.numHistograms, inputData.numHistograms);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {inputTarget.numBins, inputData.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        // find the two enclosing bins in the input histogram
        real_t outputBinPosition = inputTarget.getBinPosition(binIdx);
        idx_t leftBinIdx = inputData.getBin(outputBinPosition - 0.5_r * inputData.binSize);
        idx_t rightBinIdx = leftBinIdx + 1;

        // handle boundaries
        if (leftBinIdx < 0 || rightBinIdx >= inputData.numBins)
        {
            inputTarget.data(binIdx, histogramIdx) += 0_r;  // out of bounds, set to zero
            return;
        }

        auto inputDataLeft = inputData.data(leftBinIdx, histogramIdx);
        auto inputDataRight = inputData.data(rightBinIdx, histogramIdx);

        inputTarget.data(binIdx, histogramIdx) += lerp(
            inputDataLeft,
            inputDataRight,
            (outputBinPosition - inputData.getBinPosition(leftBinIdx)) * inputData.inverseBinSize);
    };
    Kokkos::parallel_for("MultiHistogram::updateInterpolate", policy, kernel);
    Kokkos::fence();
}

}  // namespace util
}  // namespace mrmd