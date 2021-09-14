#include "LennardJones.hpp"

#include <gtest/gtest.h>

#include <fstream>

namespace mrmd
{
namespace action
{
TEST(LennardJones, ExplicitComparison)
{
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3.01_r;
    constexpr real_t rc = 2.5_r * sigma;
    constexpr real_t cappingDistance = 0.1_r;
    real_t delta = 0.1_r;
    impl::CappedLennardJonesPotential LJ({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, true);

    const auto cutoffPotential =
        4_r * epsilon * (std::pow(sigma / rc, 12) - std::pow(sigma / rc, 6));
    for (real_t x = cappingDistance + delta; x < rc + 10 * delta; x += delta)
    {
        auto potential =
            4_r * epsilon * (std::pow(sigma / x, 12) - std::pow(sigma / x, 6)) - cutoffPotential;
        EXPECT_FLOAT_EQ(potential, LJ.computeEnergy(x * x, 0));
        auto force = 4_r * epsilon * (-12 * std::pow(sigma / x, 12) + 6 * std::pow(sigma / x, 6)) *
                     x / (x * x);
        EXPECT_FLOAT_EQ(force, -x * LJ.computeForce(x * x, 0));
    }
}

}  // namespace action
}  // namespace mrmd