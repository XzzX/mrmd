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
 *
 * @param positions
 * @param min
 * @param max
 * @param bins
 * @return
 */
ScalarView getDensityProfileX(const data::Particles::pos_t& positions,
                              idx_t numParticles,
                              real_t min,
                              real_t max,
                              int64_t bins)
{
    assert(max >= min);
    auto inverseDx = 1_r / (real_c(max - min) / real_c(bins));
    ScalarView histogram("density-profile", bins);
    ScalarScatterView scatter(histogram);
    auto policy = Kokkos::RangePolicy<>(0, numParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto bin = idx_c((positions(idx, 0) - min) * inverseDx);
        if (bin < 0) return;
        if (bin >= bins) return;
        auto access = scatter.access();
        access(bin) += 1_r;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::Experimental::contribute(histogram, scatter);
    return histogram;
}

}  // namespace analysis
}  // namespace mrmd