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
          dx((max - min) / real_c(numBins)),
          inverseDx(1_r / dx),
          data(label, numBins)
    {
    }

    const real_t min;
    const real_t max;
    const idx_t numBins;
    const real_t dx;
    const real_t inverseDx;
    ScalarView data;
};
}  // namespace data
}  // namespace mrmd