#include "LennardJones.hpp"

#include <gtest/gtest.h>

TEST(LennardJones, Equilibrium)
{
    const real_t epsilon = 2_r;
    const real_t sigma = 3_r;
    Particles particles;
    LennardJones LJ(particles, 2.5_r * sigma, sigma, epsilon);
    EXPECT_FLOAT_EQ(LJ.computeForce(std::pow(2, 1_r / 6_r) * sigma, 0_r, 0_r) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(LJ.computeForce(sigma, 0_r, 0_r), epsilon);
}
