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

#pragma once

#include <string>

#include "assert/assert.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
struct MultiHistogram
{
    MultiHistogram(const std::string& label,
                   const real_t minArg,
                   const real_t maxArg,
                   idx_t numBinsArg,
                   idx_t numHistogramsArg)
        : min(minArg),
          max(maxArg),
          numBins(numBinsArg),
          numHistograms(numHistogramsArg),
          binSize((max - min) / real_c(numBins)),
          inverseBinSize(1_r / binSize),
          data(label, numBins, numHistograms)
    {
        MRMD_HOST_CHECK_GREATER(maxArg, minArg);
        MRMD_HOST_CHECK_GREATEREQUAL(numBinsArg, 0);
        MRMD_HOST_CHECK_GREATEREQUAL(numHistogramsArg, 0);
    }

    MultiHistogram(const std::string& label, const MultiHistogram& histogram)
        : MultiHistogram(
              label, histogram.min, histogram.max, histogram.numBins, histogram.numHistograms)
    {
        Kokkos::deep_copy(data, histogram.data);
    }

    /**
     * @param val input value
     * @return corresponding bin or -1 if outside of range
     */
    KOKKOS_INLINE_FUNCTION idx_t getBin(const real_t& val) const
    {
        auto bin = idx_c((val - min) * inverseBinSize);
        if (bin < 0) bin = -1;
        if (bin >= numBins) bin = -1;
        return bin;
    }

    KOKKOS_INLINE_FUNCTION ScalarView getGrid() const
    {
        ScalarView grid("grid", numBins);
        for (auto i = 0; i < numBins; ++i)
        {
            grid[i] = min + (i + 0.5_r) * (max - min) / real_c(numBins);
        }
        return grid;
    }

    const real_t min;
    const real_t max;
    const idx_t numBins;
    const idx_t numHistograms;
    const real_t binSize;
    const real_t inverseBinSize;
    MultiView data;

    MultiHistogram& operator+=(const MultiHistogram& rhs);
    MultiHistogram& operator/=(const MultiHistogram& rhs);
    void scale(const real_t& scalingFactor);
    void scale(const ScalarView& scalingFactor);
    void makeSymmetric();
};

/**
 * Add new item to the comulative moving average.
 *
 * @param average averaged histogram (output)
 * @param current current value
 * @param movingAverageFactor new value = (n * average + current) / (n + 1)
 */
void cumulativeMovingAverage(data::MultiHistogram& average,
                             const data::MultiHistogram& current,
                             const real_t movingAverageFactor = 10_r);

/**
 * Calculates the gradient of the histogram.
 * Uses central difference for all inner values and one-sided/central difference for
 * boundary values depending on periodicity.
 *
 * @param input input histogram
 * @param periodic treat boundaries as periodic
 * @return gradient of the histogram
 */
data::MultiHistogram gradient(const data::MultiHistogram& input, const bool periodic = false);

/**
 * Smoothen a histogram using gaussian convolution.
 *
 * @param input input histogram
 * @param sigma sigma of the gaussian convolution
 * @param range calculation range in multiples of sigma
 * @param periodic treat boundaries as periodic
 * @return smoothened histogram
 */
data::MultiHistogram smoothen(data::MultiHistogram& input,
                              const real_t& sigma,
                              const real_t& range,
                              const bool periodic = false);

}  // namespace data
}  // namespace mrmd