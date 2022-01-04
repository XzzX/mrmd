#include "Random.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(Random, Range)
{
    Random rng;
    auto data = ScalarView("random-numbers", 100);
    Kokkos::parallel_for(
        "draw-numbers", Kokkos::RangePolicy<>(0, 100), KOKKOS_LAMBDA(const idx_t& idx) {
            data(idx) = rng.draw();
        });
    auto h_data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), data);
    for (auto i = 0; i < 100; ++i)
    {
        EXPECT_GE(h_data(i), 0_r);
        EXPECT_LT(h_data(i), 1_r);
    }
}
}  // namespace util
}  // namespace mrmd
