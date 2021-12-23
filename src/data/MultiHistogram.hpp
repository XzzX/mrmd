#pragma once

#include <string>

#include "assert.hpp"
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
        ASSERT_GREATER(maxArg, minArg);
        ASSERT_GREATEREQUAL(numBinsArg, 0);
        ASSERT_GREATEREQUAL(numHistogramsArg, 0);
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
 * Uses central difference for all inner values and one-sided difference for
 * boundary values.
 *
 * @param input input histogram
 * @return gradient of the histogram
 */
data::MultiHistogram gradient(const data::MultiHistogram& input);

/**
 * Smoothen a histogram using gaussian convolution.
 *
 * @param input input histogram
 * @param sigma sigma of the gaussian convolution
 * @param range calculation range in multiples of sigma
 * @return smoothened histogram
 */
data::MultiHistogram smoothen(data::MultiHistogram& input,
                              const real_t& sigma,
                              const real_t& range);

}  // namespace data
}  // namespace mrmd