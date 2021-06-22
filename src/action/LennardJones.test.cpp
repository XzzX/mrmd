#include "LennardJones.hpp"

#include <gtest/gtest.h>

TEST(LennardJones, Equilibrium)
{
    const real_t epsilon = 1_r;
    const real_t sigma = 1_r;
    Particles particles;
    LennardJones LJ(2.5_r * sigma, sigma, epsilon);
    std::cout << LJ.computeForce(2, 0, 0) << std::endl;
    EXPECT_FLOAT_EQ(LJ.computeForce(std::pow(2, 1_r / 6_r) * sigma, 0_r, 0_r) + 1_r, 1_r);
}
