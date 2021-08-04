#pragma once

#include "data/Histogram.hpp"
#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace analysis
{
data::Histogram smoothenDensityProfile(data::Histogram& densityProfile,
                                       const real_t sigma,
                                       const real_t intensity);

}  // namespace analysis
}  // namespace mrmd