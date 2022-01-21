#include "Fluctuation.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
void fluctuationNormalization()
{
    auto histogram0 = data::MultiHistogram("hist", 0_r, 1_r, 10, 1);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram0.data(idx, 0) = 3_r;
        });
    Kokkos::fence();
    auto fluctuation0 = analysis::getFluctuation(histogram0, 2_r, 0);

    auto histogram1 = data::MultiHistogram("hist", 0_r, 2_r, 10, 1);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram1.data(idx, 0) = 3_r;
        });
    Kokkos::fence();
    auto fluctuation1 = analysis::getFluctuation(histogram1, 2_r, 0);

    EXPECT_FLOAT_EQ(fluctuation0, fluctuation1);
}
TEST(Fluctuation, normalization) { fluctuationNormalization(); }

void fluctuationCheck()
{
    auto histogram = data::MultiHistogram("hist", 0_r, 2_r, 10, 2);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = 3_r;
            histogram.data(idx, 1) = 4_r;
        });
    Kokkos::fence();

    auto fluctuation0 = analysis::getFluctuation(histogram, 2_r, 0);
    auto fluctuation1 = analysis::getFluctuation(histogram, 2_r, 1);

    EXPECT_FLOAT_EQ(fluctuation0, 0.25_r);
    EXPECT_FLOAT_EQ(fluctuation1, 1_r);
}
TEST(Fluctuation, check) { fluctuationCheck(); }

void fluctuationRelation()
{
    auto histogram = data::MultiHistogram("hist", 0_r, 1_r, 10, 2);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = real_c(idx + 1);
            histogram.data(idx, 1) = real_c(idx);
        });
    Kokkos::fence();

    auto fluctuation0 = analysis::getFluctuation(histogram, 0.5_r, 0);
    auto fluctuation1 = analysis::getFluctuation(histogram, 0.5_r, 1);

    EXPECT_GT(fluctuation0, fluctuation1);
}
TEST(Fluctuation, relation) { fluctuationRelation(); }
}  // namespace analysis
}  // namespace mrmd