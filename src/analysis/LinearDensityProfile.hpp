#pragma once

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
ScalarView getLinearDensityProfile(const data::Particles::pos_t& positions,
                                   idx_t numParticles,
                                   real_t min,
                                   real_t max,
                                   int64_t bins);

}  // namespace analysis
}  // namespace mrmd