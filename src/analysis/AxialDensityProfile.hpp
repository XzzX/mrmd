#pragma once

#include "data/Atoms.hpp"
#include "data/Histogram.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
/**
 * Calculate a discretized density profile along the x axis.
 * Out-of-bounds values are discarded.
 */
data::Histogram getAxialDensityProfile(const data::Atoms::pos_t& positions,
                                       const idx_t numAtoms,
                                       const real_t min,
                                       const real_t max,
                                       const int64_t numBins);

}  // namespace analysis
}  // namespace mrmd