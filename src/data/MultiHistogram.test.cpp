#include "MultiHistogram.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace data
{
TEST(MultiHistogram, smoothen_symmetric)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    h_data(5, 0) = 10_r;
    h_data(5, 1) = 5_r;
    Kokkos::deep_copy(histogram.data, h_data);

    auto smoothedDensityProfile = smoothen(histogram, 1_r, 3_r);

    auto h_smoothed =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), smoothedDensityProfile.data);
    for (auto idx = 1; idx < 6; ++idx)
    {
        EXPECT_FLOAT_EQ(h_smoothed(5 - idx, 0), h_smoothed(5 + idx, 0));
        EXPECT_FLOAT_EQ(h_smoothed(5 - idx, 1), h_smoothed(5 + idx, 1));
    }
}

void multiHistogramSmoothen_constant()
{
    constexpr auto CONST_VAL = 2_r;
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 1);
    Kokkos::parallel_for(
        "init_histogram", Kokkos::RangePolicy<>(0, 11), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = CONST_VAL;
        });

    auto smoothedDensityProfile = smoothen(histogram, 1_r, 3_r);

    auto h_smoothed =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), smoothedDensityProfile.data);
    for (auto idx = 0; idx < 11; ++idx)
    {
        EXPECT_FLOAT_EQ(h_smoothed(idx, 0), CONST_VAL);
    }
}
TEST(MultiHistogram, smoothen_constant) { multiHistogramSmoothen_constant(); }
}  // namespace data
}  // namespace mrmd