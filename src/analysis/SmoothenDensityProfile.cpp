#include "SmoothenDensityProfile.hpp"

namespace mrmd
{
namespace analysis
{
ScalarView smoothenDensityProfile(ScalarView& densityProfile,
                                  const real_t binSize,
                                  const real_t sigma,
                                  const real_t inten)
{
    const idx_t numBins = idx_c(densityProfile.extent(0));

    const auto inverseBinSize = 1_r / binSize;
    const auto inverseSigma = 1_r / sigma;
    /// how many neighboring bins are affected
    const idx_t delta = int_c(inten * sigma * inverseBinSize);

    ScalarView smoothenedDensityProfile("smooth-density-profile", densityProfile.extent(0));

    auto policy = Kokkos::RangePolicy<>(0, numBins);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto normalization = 0_r;

        const idx_t jdxMin = std::max(idx_t(0), idx - delta);
        const idx_t jdxMax = std::min(numBins - 1, idx + delta);

        for (auto jdx = jdxMin; jdx <= jdxMax; ++jdx)
        {
            const auto eFunc = std::exp(-util::sqr(real_c(idx - jdx) * binSize * inverseSigma));
            normalization += eFunc * binSize;
            smoothenedDensityProfile(idx) += densityProfile(jdx) * eFunc * binSize;
        }

        smoothenedDensityProfile(idx) /= normalization;
    };
    Kokkos::parallel_for(policy, kernel);

    return smoothenedDensityProfile;
}

}  // namespace analysis
}  // namespace mrmd