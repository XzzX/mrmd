#include "SmoothenDensityProfile.hpp"

namespace mrmd
{
namespace analysis
{
data::MultiHistogram smoothenDensityProfile(data::MultiHistogram& densityProfile,
                                            const real_t sigma,
                                            const real_t intensity)
{
    const auto inverseSigma = 1_r / sigma;
    /// how many neighboring bins are affected
    const idx_t delta = int_c(intensity * sigma * densityProfile.inverseBinSize);

    data::MultiHistogram smoothenedDensityProfile("smooth-density-profile",
                                                  densityProfile.min,
                                                  densityProfile.max,
                                                  densityProfile.numBins,
                                                  densityProfile.numHistograms);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {densityProfile.numBins, densityProfile.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        auto normalization = 0_r;

        const idx_t jdxMin = std::max(idx_t(0), binIdx - delta);
        const idx_t jdxMax = std::min(densityProfile.numBins - 1, binIdx + delta);
        assert(jdxMin <= jdxMax);

        for (auto jdx = jdxMin; jdx <= jdxMax; ++jdx)
        {
            const auto eFunc =
                std::exp(-util::sqr(real_c(binIdx - jdx) * densityProfile.binSize * inverseSigma));
            normalization += eFunc;
            smoothenedDensityProfile.data(binIdx, histogramIdx) +=
                densityProfile.data(jdx, histogramIdx) * eFunc;
        }

        smoothenedDensityProfile.data(binIdx, histogramIdx) /= normalization;
    };
    Kokkos::parallel_for(policy, kernel);

    return smoothenedDensityProfile;
}

}  // namespace analysis
}  // namespace mrmd