#include "SystemMomentum.hpp"

#include <gtest/gtest.h>

#include "data/Particles.hpp"

TEST(SystemMomentum, Simple)
{
    Particles particles(2);
    auto vel = particles.getVel();
    vel(0, 0) = +2_r;
    vel(0, 1) = +3_r;
    vel(0, 2) = +4_r;
    vel(1, 0) = -4_r;
    vel(1, 1) = -8_r;
    vel(1, 2) = -16_r;
    particles.numLocalParticles = 2;
    particles.numGhostParticles = 0;

    auto systemMomentum = analysis::getSystemMomentum(particles);

    EXPECT_FLOAT_EQ(systemMomentum[0], -2_r);
    EXPECT_FLOAT_EQ(systemMomentum[1], -5_r);
    EXPECT_FLOAT_EQ(systemMomentum[2], -12_r);
}