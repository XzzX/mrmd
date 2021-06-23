#include "LennardJones.hpp"

#include <gtest/gtest.h>

TEST(LennardJones, Force)
{
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3_r;
    constexpr real_t rc = 2.5_r * sigma;
    constexpr real_t eps = 0.001_r;
    real_t minimumSqr = std::pow(2, 1_r / 6_r) * sigma;
    minimumSqr *= minimumSqr;
    LennardJones LJ(rc, sigma, epsilon);

    EXPECT_GT(LJ.computeForce_(minimumSqr - eps) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(LJ.computeForce_(minimumSqr) + 1_r, 1_r);
    EXPECT_LT(LJ.computeForce_(minimumSqr + eps) + 1_r, 1_r);
}

TEST(LennardJones, Energy)
{
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3_r;
    constexpr real_t rc = 2.5_r * sigma;
    real_t minimumSqr = std::pow(2, 1_r / 6_r) * sigma;
    minimumSqr *= minimumSqr;
    LennardJones LJ(rc, sigma, epsilon);

    EXPECT_FLOAT_EQ(LJ.computeEnergy_(sigma * sigma), 0_r);
    EXPECT_FLOAT_EQ(LJ.computeEnergy_(minimumSqr), -epsilon);
}
