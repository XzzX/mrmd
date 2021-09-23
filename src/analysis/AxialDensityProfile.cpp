#include "AxialDensityProfile.hpp"

namespace mrmd
{
namespace analysis
{
data::Histogram getAxialDensityProfile(const data::Atoms::pos_t& positions,
                                       const idx_t numAtoms,
                                       const real_t min,
                                       const real_t max,
                                       const int64_t numBins)
{
    assert(max >= min);
    data::Histogram histogram("density-profile", min, max, numBins);
    ScalarScatterView scatter(histogram.data);

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto bin = histogram.getBin(positions(idx, COORD_X));
        if (bin == -1) return;
        auto access = scatter.access();
        access(bin) += 1_r;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::Experimental::contribute(histogram.data, scatter);
    Kokkos::fence();

    return histogram;
}

}  // namespace analysis
}  // namespace mrmd