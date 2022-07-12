#include "Fluctuation.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
void fluctuationNormalization()
{
    auto histogram0 = data::MultiHistogram("hist", real_t(0), real_t(1), 10, 1);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram0.data(idx, 0) = real_t(3);
        });
    Kokkos::fence();
    auto fluctuation0 = analysis::getFluctuation(histogram0, real_t(2), 0);

    auto histogram1 = data::MultiHistogram("hist", real_t(0), real_t(2), 10, 1);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram1.data(idx, 0) = real_t(3);
        });
    Kokkos::fence();
    auto fluctuation1 = analysis::getFluctuation(histogram1, real_t(2), 0);

    EXPECT_FLOAT_EQ(fluctuation0, fluctuation1);
}
TEST(Fluctuation, normalization) { fluctuationNormalization(); }

void fluctuationCheck()
{
    auto histogram = data::MultiHistogram("hist", real_t(0), real_t(2), 10, 2);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = real_t(3);
            histogram.data(idx, 1) = real_t(4);
        });
    Kokkos::fence();

    auto fluctuation0 = analysis::getFluctuation(histogram, real_t(2), 0);
    auto fluctuation1 = analysis::getFluctuation(histogram, real_t(2), 1);

    EXPECT_FLOAT_EQ(fluctuation0, real_t(0.25));
    EXPECT_FLOAT_EQ(fluctuation1, real_t(1));
}
TEST(Fluctuation, check) { fluctuationCheck(); }

void fluctuationRelation()
{
    auto histogram = data::MultiHistogram("hist", real_t(0), real_t(1), 10, 2);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = real_c(idx + 1);
            histogram.data(idx, 1) = real_c(idx);
        });
    Kokkos::fence();

    auto fluctuation0 = analysis::getFluctuation(histogram, real_t(0.5), 0);
    auto fluctuation1 = analysis::getFluctuation(histogram, real_t(0.5), 1);

    EXPECT_GT(fluctuation0, fluctuation1);
}
TEST(Fluctuation, relation) { fluctuationRelation(); }
}  // namespace analysis
}  // namespace mrmd