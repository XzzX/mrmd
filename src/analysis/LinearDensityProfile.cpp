#include "LinearDensityProfile.hpp"

namespace mrmd
{
namespace analysis
{
data::Histogram getLinearDensityProfile(const data::Particles::pos_t& positions,
                                        const idx_t numParticles,
                                        const real_t min,
                                        const real_t max,
                                        const int64_t numBins)
{
    assert(max >= min);
    data::Histogram histogram("density-profile", min, max, numBins);
    ScalarScatterView scatter(histogram.data);

    auto policy = Kokkos::RangePolicy<>(0, numParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto bin = idx_c((positions(idx, COORD_X) - min) * histogram.inverseBinSize);
        if (bin < 0) return;
        if (bin >= histogram.numBins) return;
        auto access = scatter.access();
        access(bin) += 1_r;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::Experimental::contribute(histogram.data, scatter);

    return histogram;
}

}  // namespace analysis
}  // namespace mrmd