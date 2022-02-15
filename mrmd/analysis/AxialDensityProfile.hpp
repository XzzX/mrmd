#pragma once

#include <vector>

#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
/**
 * Calculate a discretized density profile along the x axis.
 * Out-of-bounds values are discarded.
 */
data::MultiHistogram getAxialDensityProfile(const idx_t numAtoms,
                                            const data::Atoms::pos_t& positions,
                                            const data::Atoms::type_t& types,
                                            const int64_t numTypes,
                                            const real_t min,
                                            const real_t max,
                                            const int64_t numBins,
                                            const int64_t axis);

}  // namespace analysis
}  // namespace mrmd