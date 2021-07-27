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

    const real_t min;
    const real_t max;
    const idx_t numBins;
    const real_t binSize;
    const real_t inverseBinSize;
    ScalarView data;
};
}  // namespace data
}  // namespace mrmd