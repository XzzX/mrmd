#include "SmoothenDensityProfile.hpp"

namespace mrmd
{
namespace analysis
{
data::Histogram smoothenDensityProfile(data::Histogram& densityProfile,
                                       const real_t sigma,
                                       const real_t intensity)
{
    const auto inverseSigma = 1_r / sigma;
    /// how many neighboring bins are affected
    const idx_t delta = int_c(intensity * sigma * densityProfile.inverseBinSize);

    data::Histogram smoothenedDensityProfile(
        "smooth-density-profile", densityProfile.min, densityProfile.max, densityProfile.numBins);

    auto policy = Kokkos::RangePolicy<>(0, densityProfile.numBins);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto normalization = 0_r;

        const idx_t jdxMin = std::max(idx_t(0), idx - delta);
        const idx_t jdxMax = std::min(densityProfile.numBins - 1, idx + delta);
        assert(jdxMin <= jdxMax);

        for (auto jdx = jdxMin; jdx <= jdxMax; ++jdx)
        {
            const auto eFunc =
                std::exp(-util::sqr(real_c(idx - jdx) * densityProfile.binSize * inverseSigma));
            normalization += eFunc;
            smoothenedDensityProfile.data(idx) += densityProfile.data(jdx) * eFunc;
        }

        smoothenedDensityProfile.data(idx) /= normalization;
    };
    Kokkos::parallel_for(policy, kernel);

    return smoothenedDensityProfile;
}

}  // namespace analysis
}  // namespace mrmd