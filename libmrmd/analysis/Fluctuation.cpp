#include "Fluctuation.hpp"

#include "util/math.hpp"

namespace mrmd
{
namespace analysis
{
real_t getFluctuation(const data::MultiHistogram& hist,
                      const real_t& reference,
                      const idx_t& specimen)
{
    auto ret = 0_r;
    const real_t weighting = hist.binSize / (hist.max - hist.min);
    auto data = hist.data;

    auto policy = Kokkos::RangePolicy<>(0, hist.numBins);
    auto normalizeSampleKernel = KOKKOS_LAMBDA(const idx_t& idx, real_t& fluctuation)
    {
        fluctuation += weighting * util::sqr((data(idx, specimen) - reference) / reference);
    };
    Kokkos::parallel_reduce("getFluctuation", policy, normalizeSampleKernel, ret);
    Kokkos::fence();

    return ret;
}

}  // namespace analysis
}  // namespace mrmd