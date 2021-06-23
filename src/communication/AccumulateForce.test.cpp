#include "communication/AccumulateForce.hpp"

#include <gtest/gtest.h>

#include <data/Particles.hpp>

TEST(AccumulateForceTest, ghostToReal)
{
    // accumulate all force on particle 0
    Particles particles(101);
    particles.numLocalParticles = 1;
    particles.numGhostParticles = 100;
    particles.resize(101);

    auto ghost = particles.getGhost();
    Cabana::deep_copy(ghost, 0);
    ghost(0) = -1;

    auto force = particles.getForce();
    Cabana::deep_copy(force, 1_r);

    AccumulateForce accumulateForce;
    accumulateForce.ghostToReal(particles);

    EXPECT_FLOAT_EQ(force(0, 0), 101_r);
    EXPECT_FLOAT_EQ(force(0, 1), 101_r);
    EXPECT_FLOAT_EQ(force(0, 2), 101_r);
}
