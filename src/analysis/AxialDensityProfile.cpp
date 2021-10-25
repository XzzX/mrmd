#include "AxialDensityProfile.hpp"

#include <fmt/format.h>

namespace mrmd
{
namespace analysis
{
std::vector<data::Histogram> getAxialDensityProfile(const idx_t numAtoms,
                                                    const data::Atoms::pos_t& positions,
                                                    const data::Atoms::type_t& types,
                                                    const int64_t numTypes,
                                                    const real_t min,
                                                    const real_t max,
                                                    const int64_t numBins)
{
    assert(max >= min);
    assert(numTypes > 0);
    std::vector<data::Histogram> histogram;
    std::vector<ScalarScatterView> scatter;
    for (auto i = 0; i < numTypes; ++i)
    {
        histogram.emplace_back(fmt::format("density-profile-{}", i), min, max, numBins);
        scatter.emplace_back(histogram[i].data);
    }

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        assert(types(idx) >= 0);
        assert(types(idx) < numTypes);
        auto bin = histogram[types(idx)].getBin(positions(idx, COORD_X));
        if (bin == -1) return;
        auto access = scatter[types(idx)].access();
        access(bin) += 1_r;
    };
    Kokkos::parallel_for(policy, kernel);
    for (auto i = 0; i < numTypes; ++i)
    {
        Kokkos::Experimental::contribute(histogram[i].data, scatter[i]);
    }
    Kokkos::fence();

    return histogram;
}

}  // namespace analysis
}  // namespace mrmd