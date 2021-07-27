#pragma once

#include "data/Histogram.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
/**
 * Calculate a discretized density profile along the x axis.
 * Out-of-bounds values are discarded.
 */
data::Histogram getLinearDensityProfile(const data::Particles::pos_t& positions,
                                        const idx_t numParticles,
                                        const real_t min,
                                        const real_t max,
                                        const int64_t numBins);

}  // namespace analysis
}  // namespace mrmd