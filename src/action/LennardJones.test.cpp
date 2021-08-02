#include "LennardJones.hpp"

#include <gtest/gtest.h>

#include <fstream>

namespace mrmd
{
namespace action
{
TEST(LennardJones, Force)
{
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3_r;
    constexpr real_t rc = 2.5_r * sigma;
    constexpr real_t cappingDistance = 0_r;
    constexpr real_t eps = 0.001_r;
    real_t minimumSqr = std::pow(2, 1_r / 6_r) * sigma;
    minimumSqr *= minimumSqr;
    impl::CappedLennardJonesPotential LJ(cappingDistance, rc, sigma, epsilon, false);

    EXPECT_GT(LJ.computeForce(minimumSqr - eps) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(LJ.computeForce(minimumSqr) + 1_r, 1_r);
    EXPECT_LT(LJ.computeForce(minimumSqr + eps) + 1_r, 1_r);

    EXPECT_FLOAT_EQ(LJ.computeForce(sigma * sigma), 8_r * epsilon / sigma);
}

TEST(LennardJones, direction)
{
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3_r;
    constexpr real_t rc = 2.5_r * sigma;
    constexpr real_t cappingDistance = 0.5_r * sigma;
    constexpr real_t eps = 0.001_r;
    real_t minimum = std::pow(2, 1_r / 6_r) * sigma;
    real_t minimumSqr = minimum * minimum;
    impl::CappedLennardJonesPotential LJ(cappingDistance, rc, sigma, epsilon, false);

    real_t previousValue = std::numeric_limits<real_t>::max();
    for (real_t x = 0.1_r; x < minimum; x += 0.1_r)
    {
        auto tmp = LJ.computeForce(x * x) * x;
        EXPECT_LE(tmp, previousValue);  // monotonously decaying
        EXPECT_GT(tmp, 0_r);            // repulsive force
        previousValue = tmp;
    }

    previousValue = LJ.computeForce(minimumSqr) * minimum;
    for (real_t x = minimum + 0.1_r; x < rc; x += 0.1_r)
    {
        std::cout << x << std::endl;
        auto tmp = LJ.computeForce(x * x) * x;
        EXPECT_LT(tmp, 0_r);  // attractive force
        previousValue = tmp;
    }
}

TEST(LennardJones, Energy)
{
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3_r;
    constexpr real_t rc = 2.5_r * sigma;
    constexpr real_t cappingDistance = 0_r;
    real_t minimumSqr = std::pow(2, 1_r / 6_r) * sigma;
    minimumSqr *= minimumSqr;
    impl::CappedLennardJonesPotential LJUnshifted(cappingDistance, rc, sigma, epsilon, false);
    auto rcEnergy = LJUnshifted.computeEnergy(rc * rc);
    EXPECT_FLOAT_EQ(LJUnshifted.computeEnergy(sigma * sigma), 0_r);
    EXPECT_FLOAT_EQ(LJUnshifted.computeEnergy(minimumSqr), -epsilon);

    impl::CappedLennardJonesPotential LJShifted(cappingDistance, rc, sigma, epsilon, true);
    EXPECT_FLOAT_EQ(LJShifted.computeEnergy(sigma * sigma), -rcEnergy);
    EXPECT_FLOAT_EQ(LJShifted.computeEnergy(rc * rc), 0_r);
    EXPECT_FLOAT_EQ(LJShifted.computeEnergy(minimumSqr), -epsilon - rcEnergy);
}

TEST(LennardJones, ExplicitComparison)
{
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3.01_r;
    constexpr real_t rc = 2.5_r * sigma;
    constexpr real_t cappingDistance = 0.1_r;
    real_t delta = 0.1_r;
    impl::CappedLennardJonesPotential LJ(cappingDistance, rc, sigma, epsilon, true);

    const auto cutoffPotential =
        4_r * epsilon * (std::pow(sigma / rc, 12) - std::pow(sigma / rc, 6));
    for (real_t x = cappingDistance + delta; x < rc + 10 * delta; x += delta)
    {
        auto potential =
            4_r * epsilon * (std::pow(sigma / x, 12) - std::pow(sigma / x, 6)) - cutoffPotential;
        EXPECT_FLOAT_EQ(potential, LJ.computeEnergy(x * x));
        auto force = 4_r * epsilon * (-12 * std::pow(sigma / x, 12) + 6 * std::pow(sigma / x, 6)) *
                     x / (x * x);
        EXPECT_FLOAT_EQ(force, -x * LJ.computeForce(x * x));
    }
}

}  // namespace action
}  // namespace mrmd