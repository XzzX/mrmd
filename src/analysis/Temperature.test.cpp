#include "Temperature.hpp"

#include <gtest/gtest.h>

#include "data/Particles.hpp"

TEST(Temperature, Simple)
{
    data::Particles particles(3);
    auto vel = particles.getVel();
    vel(0, 0) = +2_r;
    vel(0, 1) = +0_r;
    vel(0, 2) = +0_r;
    vel(1, 0) = -0_r;
    vel(1, 1) = -8_r;
    vel(1, 2) = -0_r;
    vel(2, 0) = +0_r;
    vel(2, 1) = +0_r;
    vel(2, 2) = +16_r;
    particles.numLocalParticles = 3;
    particles.numGhostParticles = 0;

    auto temperature = analysis::getTemperature(particles);

    EXPECT_FLOAT_EQ(temperature, (4_r + 64_r + 256_r) / 9_r);
}