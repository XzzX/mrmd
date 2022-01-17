#include "AxialDensityProfile.hpp"

#include <fmt/format.h>

namespace mrmd
{
namespace analysis
{
data::MultiHistogram getAxialDensityProfile(const idx_t numAtoms,
                                            const data::Atoms::pos_t& positions,
                                            const data::Atoms::type_t& types,
                                            const int64_t numTypes,
                                            const real_t min,
                                            const real_t max,
                                            const int64_t numBins,
                                            const int64_t axis)
{
    MRMD_HOST_CHECK_GREATEREQUAL(max, min);
    MRMD_HOST_CHECK_GREATER(numTypes, 0);
    MRMD_HOST_CHECK_GREATEREQUAL(axis, 0);
    MRMD_HOST_CHECK_LESSEQUAL(axis, 3);

    data::MultiHistogram histogram("density-profile", min, max, numBins, numTypes);
    MultiScatterView scatter(histogram.data);

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        MRMD_DEVICE_ASSERT_GREATEREQUAL(types(idx), 0);
        MRMD_DEVICE_ASSERT_LESS(types(idx), numTypes);
        auto bin = histogram.getBin(positions(idx, axis));
        if (bin == -1) return;
        auto access = scatter.access();
        access(bin, types(idx)) += 1_r;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::Experimental::contribute(histogram.data, scatter);
    Kokkos::fence();

    return histogram;
}

}  // namespace analysis
}  // namespace mrmd