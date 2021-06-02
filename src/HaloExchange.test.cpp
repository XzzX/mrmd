#include "HaloExchange.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "Subdomain.hpp"

TEST(HaloExchange, SelfExchange)
{
    auto subdomain = Subdomain({0_r, 0_r, 0_r}, {2_r, 2_r, 2_r}, 0.7_r);

    Particles particles;
    int64_t idx = 0;
    for (double x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x+=1_r)
        for (double y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y+=1_r)
            for (double z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2]; z+=1_r)
            {
                particles.getPos(idx, 0) = x;
                particles.getPos(idx, 1) = y;
                particles.getPos(idx, 2) = z;
                ++idx;
            }
    EXPECT_EQ(idx, 8);
    particles.numLocalParticles = 8;
    particles.numGhostParticles = 0;

    auto haloExchange = HaloExchange(particles, subdomain);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagX>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 8);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagY>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 24);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagZ>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 56);
}