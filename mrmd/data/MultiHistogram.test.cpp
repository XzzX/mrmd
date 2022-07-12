#include "MultiHistogram.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace data
{

TEST(MultiHistogram, scale)
{
    MultiHistogram histogram("histogram", real_t(0), real_t(10), 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    h_data(5, 0) = real_t(10);
    h_data(5, 1) = real_t(5);
    Kokkos::deep_copy(histogram.data, h_data);

    histogram.scale(real_t(3));

    Kokkos::deep_copy(h_data, histogram.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_data(idx, 0), idx == 5 ? real_t(30) : real_t(0));
        EXPECT_FLOAT_EQ(h_data(idx, 1), idx == 5 ? real_t(15) : real_t(0));
    }
}

TEST(MultiHistogram, make_symmetric)
{
    MultiHistogram histogram("histogram", real_t(0), real_t(10), 10, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    for (auto idx = 0; idx < 10; ++idx)
    {
        h_data(idx, 0) = real_c(idx);
        h_data(idx, 1) = real_t(10) - real_c(idx);
    }

    Kokkos::deep_copy(histogram.data, h_data);

    histogram.makeSymmetric();

    Kokkos::deep_copy(h_data, histogram.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_data(idx, 0), real_t(4.5));
        EXPECT_FLOAT_EQ(h_data(idx, 1), real_t(5.5));
    }
}

TEST(MultiHistogram, op_plusequal)
{
    MultiHistogram histogram("histogram", real_t(0), real_t(10), 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    h_data(5, 0) = real_t(10);
    h_data(5, 1) = real_t(5);
    Kokkos::deep_copy(histogram.data, h_data);

    histogram += histogram;

    Kokkos::deep_copy(h_data, histogram.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_data(idx, 0), idx == 5 ? real_t(20) : real_t(0));
        EXPECT_FLOAT_EQ(h_data(idx, 1), idx == 5 ? real_t(10) : real_t(0));
    }
}

TEST(MultiHistogram, op_divequal)
{
    MultiHistogram histogram("histogram", real_t(0), real_t(10), 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    for (auto idx = 0; idx < 10; ++idx)
    {
        h_data(idx, 0) = -real_c(idx) - real_t(1);
        h_data(idx, 1) = real_c(idx) + real_t(1);
    }
    Kokkos::deep_copy(histogram.data, h_data);

    histogram /= histogram;

    Kokkos::deep_copy(h_data, histogram.data);

    for (auto idx = 0; idx < 10; ++idx)
    {
        EXPECT_FLOAT_EQ(h_data(idx, 0), real_t(1));
        EXPECT_FLOAT_EQ(h_data(idx, 1), real_t(1));
    }
}

TEST(MultiHistogram, smoothen_symmetric)
{
    MultiHistogram histogram("histogram", real_t(0), real_t(10), 11, 2);
    auto h_data = Kokkos::create_mirror_view(histogram.data);
    h_data(5, 0) = real_t(10);
    h_data(5, 1) = real_t(5);
    Kokkos::deep_copy(histogram.data, h_data);

    auto smoothedDensityProfile = smoothen(histogram, real_t(1), real_t(3));

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
    constexpr auto CONST_VAL = real_t(2);
    MultiHistogram histogram("histogram", real_t(0), real_t(10), 11, 1);
    Kokkos::parallel_for(
        "init_histogram", Kokkos::RangePolicy<>(0, 11), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = CONST_VAL;
        });

    auto smoothedDensityProfile = smoothen(histogram, real_t(1), real_t(3));

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