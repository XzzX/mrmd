#include "PeriodicMapping.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "Subdomain.hpp"

void testMapping(const std::array<real_t, 3>& initialPos, const std::array<real_t, 3>& mappedPos)
{
    Subdomain subdomain = Subdomain({0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, 0_r);
    Particles particles;
    particles.numLocalParticles = 1;
    particles.getPos(0, 0) = initialPos[0];
    particles.getPos(0, 1) = initialPos[1];
    particles.getPos(0, 2) = initialPos[2];
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, particles.numLocalParticles),
                         PeriodicMapping(particles, subdomain));
    EXPECT_FLOAT_EQ(particles.getPos(0, 0), mappedPos[0]);
    EXPECT_FLOAT_EQ(particles.getPos(0, 1), mappedPos[1]);
    EXPECT_FLOAT_EQ(particles.getPos(0, 2), mappedPos[2]);
}

TEST(PeriodicMapping, inside)
{
    testMapping({0.4_r,0.5_r,0.6_r}, {0.4_r,0.5_r,0.6_r});
}

TEST(PeriodicMapping, periodicX)
{
    testMapping({1.1_r,0.5_r,0.6_r}, {0.1_r,0.5_r,0.6_r});
    testMapping({-0.1_r,0.5_r,0.6_r}, {0.9_r,0.5_r,0.6_r});
}

TEST(PeriodicMapping, periodicY)
{
    testMapping({0.4_r,1.1_r,0.6_r}, {0.4_r,0.1_r,0.6_r});
    testMapping({0.4_r,-0.1_r,0.6_r}, {0.4_r,0.9_r,0.6_r});
}

TEST(PeriodicMapping, periodicZ)
{
    testMapping({0.4_r,0.5_r,1.1_r}, {0.4_r,0.5_r,0.1_r});
    testMapping({0.4_r,0.5_r,-0.1_r}, {0.4_r,0.5_r,0.9_r});
}

TEST(PeriodicMapping, periodicXYZ)
{
    testMapping({1.1_r,1.2_r,1.3_r}, {0.1_r,0.2_r,0.3_r});
    testMapping({-0.3_r,-0.2_r,-0.1_r}, {0.7_r,0.8_r,0.9_r});
}