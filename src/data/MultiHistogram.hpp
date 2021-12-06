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
};

data::MultiHistogram gradient(const data::MultiHistogram& input);

}  // namespace data
}  // namespace mrmd