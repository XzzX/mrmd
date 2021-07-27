#include "LinearDensityProfile.hpp"

namespace mrmd
{
namespace analysis
{
ScalarView getLinearDensityProfile(const data::Particles::pos_t& positions,
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
        auto bin = idx_c((positions(idx, COORD_X) - min) * inverseDx);
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