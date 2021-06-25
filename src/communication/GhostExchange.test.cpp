#include "GhostExchange.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "data/Subdomain.hpp"

size_t countWithinCutoff(Particles particles,
                         const real_t cutoff,
                         const double* box,
                         const bool periodic)
{
    auto numLocalParticles = particles.numLocalParticles;
    auto numGhostParticles = particles.numGhostParticles;
    auto rcSqr = cutoff * cutoff;
    auto pos = particles.getPos();
    real_t box_diameter[3] = {box[0], box[1], box[2]};

    size_t count = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, particles.numLocalParticles),
        KOKKOS_LAMBDA(const idx_t idx, size_t& sum)
        {
            for (auto jdx = idx + 1; jdx < numLocalParticles + numGhostParticles; ++jdx)
            {
                auto dx = std::abs(pos(idx, 0) - pos(jdx, 0));
                if (periodic && (dx > box_diameter[0] * 0.5_r)) dx -= box_diameter[0];
                auto dy = std::abs(pos(idx, 1) - pos(jdx, 1));
                if (periodic && (dy > box_diameter[1] * 0.5_r)) dy -= box_diameter[1];
                auto dz = std::abs(pos(idx, 2) - pos(jdx, 2));
                if (periodic && (dz > box_diameter[2] * 0.5_r)) dz -= box_diameter[2];
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

class GhostExchangeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto pos = particles.getPos();
        int64_t idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    pos(idx, 0) = x;
                    pos(idx, 1) = y;
                    pos(idx, 2) = z;
                    ++idx;
                }
        EXPECT_EQ(idx, 27);
        particles.numLocalParticles = 27;
        particles.numGhostParticles = 0;
        auto ghost = particles.getGhost();
        Cabana::deep_copy(ghost, idx_c(-1));
    }

    // void TearDown() override {}

    Subdomain subdomain = Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    Particles particles = Particles(200);
};

TEST_F(GhostExchangeTest, SelfExchangeX)
{
    EXPECT_EQ(particles.numGhostParticles, 0);
    auto ghostExchange = impl::GhostExchange(subdomain);
    ghostExchange.exchangeGhosts<impl::GhostExchange::DIRECTION_X>(particles);
    EXPECT_EQ(particles.numGhostParticles, 18);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(particles.getGhost()(idx), -1);
        }
        else
        {
            EXPECT_LT(particles.getGhost()(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeY)
{
    auto ghostExchange = impl::GhostExchange(subdomain);
    ghostExchange.exchangeGhosts<impl::GhostExchange::DIRECTION_Y>(particles);
    EXPECT_EQ(particles.numGhostParticles, 18);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(particles.getGhost()(idx), -1);
        }
        else
        {
            EXPECT_LT(particles.getGhost()(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeZ)
{
    auto ghostExchange = impl::GhostExchange(subdomain);
    ghostExchange.exchangeGhosts<impl::GhostExchange::DIRECTION_Z>(particles);
    EXPECT_EQ(particles.numGhostParticles, 18);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(particles.getGhost()(idx), -1);
        }
        else
        {
            EXPECT_LT(particles.getGhost()(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeXYZ)
{
    auto ghostExchange = GhostExchange(subdomain);
    ghostExchange.exchangeGhostsXYZ(particles);
    EXPECT_EQ(particles.numGhostParticles, 98);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(particles.getGhost()(idx), -1);
        }
        else
        {
            EXPECT_LT(particles.getGhost()(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, CountPairs)
{
    size_t numPairs = 0;
    numPairs = countWithinCutoff(particles, 1.1_r, subdomain.diameter.data(), false);
    EXPECT_EQ(numPairs, 12 * 3 + 9 * 2);

    numPairs = countWithinCutoff(particles, 1.1_r, subdomain.diameter.data(), true);
    EXPECT_EQ(numPairs, 27 * 6 / 2);

    auto ghostExchange = GhostExchange(subdomain);
    ghostExchange.exchangeGhostsXYZ(particles);
    EXPECT_EQ(particles.numGhostParticles, 98);

    numPairs = countWithinCutoff(particles, 1.1_r, subdomain.diameter.data(), false);
    EXPECT_EQ(numPairs, 108);
}