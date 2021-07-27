#pragma once

#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace analysis
{
ScalarView smoothenDensityProfile(ScalarView& densityProfile,
                                  const real_t binSize,
                                  const real_t sigma,
                                  const real_t inten);

}  // namespace analysis
}  // namespace mrmd