#include "HaloExchange.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "Subdomain.hpp"

size_t countWithinCutoff(Particles& particles,
                         const real_t& cutoff,
                         const double* box,
                         const bool periodic)
{
    auto rcSqr = cutoff * cutoff;
    auto pos = particles.getPos();

    size_t count = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::Serial>(0, particles.numLocalParticles),
        KOKKOS_LAMBDA(const idx_t idx, size_t& sum)
        {
            for (auto jdx = idx + 1;
                 jdx < particles.numLocalParticles + particles.numGhostParticles;
                 ++jdx)
            {
                auto dx = std::abs(pos(idx, 0) - pos(jdx, 0));
                if (periodic && (dx > box[0] * 0.5_r)) dx -= box[0];
                auto dy = std::abs(pos(idx, 1) - pos(jdx, 1));
                if (periodic && (dy > box[1] * 0.5_r)) dy -= box[1];
                auto dz = std::abs(pos(idx, 2) - pos(jdx, 2));
                if (periodic && (dz > box[2] * 0.5_r)) dz -= box[2];
                auto distSqr = dx * dx + dy * dy + dz * dz;
                if (distSqr < rcSqr)
                {
                    ++sum;
                }
            }
        },
        count);

    return count;
}

class HaloExchangeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int64_t idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    particles.getPos(idx, 0) = x;
                    particles.getPos(idx, 1) = y;
                    particles.getPos(idx, 2) = z;
                    ++idx;
                }
        EXPECT_EQ(idx, 27);
        particles.numLocalParticles = 27;
        particles.numGhostParticles = 0;
    }

    // void TearDown() override {}

    Subdomain subdomain = Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    Particles particles;
};

TEST_F(HaloExchangeTest, SelfExchangeX)
{
    auto haloExchange = HaloExchange(particles, subdomain);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagX>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 18);
}

TEST_F(HaloExchangeTest, SelfExchangeY)
{
    auto haloExchange = HaloExchange(particles, subdomain);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagY>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 18);
}

TEST_F(HaloExchangeTest, SelfExchangeZ)
{
    auto haloExchange = HaloExchange(particles, subdomain);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagZ>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 18);
}

TEST_F(HaloExchangeTest, SelfExchangeXYZ)
{
    auto haloExchange = HaloExchange(particles, subdomain);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagX>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 18);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagY>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 48);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagZ>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 98);
}

TEST_F(HaloExchangeTest, CountPairs)
{
    size_t numPairs = 0;
    numPairs = countWithinCutoff(particles, 1.1_r, subdomain.diameter.data(), false);
    EXPECT_EQ(numPairs, 12 * 3 + 9 * 2);

    numPairs = countWithinCutoff(particles, 1.1_r, subdomain.diameter.data(), true);
    EXPECT_EQ(numPairs, 27 * 6 / 2);

    auto haloExchange = HaloExchange(particles, subdomain);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagX>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 18);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagY>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 48);
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagZ>(
                             0, particles.numLocalParticles + particles.numGhostParticles),
                         haloExchange);
    EXPECT_EQ(particles.numGhostParticles, 98);

    numPairs = countWithinCutoff(particles, 1.1_r, subdomain.diameter.data(), false);
    EXPECT_EQ(numPairs, 108);
}