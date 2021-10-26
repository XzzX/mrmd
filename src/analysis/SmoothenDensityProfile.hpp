#pragma once

#include "data/MultiHistogram.hpp"
#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace analysis
{
data::MultiHistogram smoothenDensityProfile(data::MultiHistogram& densityProfile,
                                            const real_t sigma,
                                            const real_t intensity);

}  // namespace analysis
}  // namespace mrmd