#pragma once

#include <string>

#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
struct Histogram
{
    Histogram(const std::string& label, const real_t minArg, const real_t maxArg, idx_t numBinsArg)
        : min(minArg),
          max(maxArg),
          numBins(numBinsArg),
          binSize((max - min) / real_c(numBins)),
          inverseBinSize(1_r / binSize),
          data(label, numBins)
    {
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
    const real_t binSize;
    const real_t inverseBinSize;
    ScalarView data;

    Histogram& operator+=(const Histogram& rhs);
};

data::Histogram gradient(const data::Histogram& input);

}  // namespace data
}  // namespace mrmd