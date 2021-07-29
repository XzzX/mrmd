#include "GhostExchange.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
size_t countWithinCutoff(const data::Particles& particles,
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
        particles.resize(particles.numLocalParticles + particles.numGhostParticles);
    }

    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    data::Particles particles = data::Particles(200);
};

TEST_F(GhostExchangeTest, SelfExchangeXHigh)
{
    EXPECT_EQ(particles.numGhostParticles, 0);
    auto ghostExchange = GhostExchange(subdomain);
    auto correspondingRealParticle = ghostExchange.exchangeGhosts<GhostExchange::DIRECTION_X_HIGH>(
        particles, particles.numLocalParticles);
    EXPECT_EQ(particles.numGhostParticles, 9);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(correspondingRealParticle(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealParticle(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeXLow)
{
    EXPECT_EQ(particles.numGhostParticles, 0);
    auto ghostExchange = GhostExchange(subdomain);
    auto correspondingRealParticle = ghostExchange.exchangeGhosts<GhostExchange::DIRECTION_X_LOW>(
        particles, particles.numLocalParticles);
    EXPECT_EQ(particles.numGhostParticles, 9);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(correspondingRealParticle(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealParticle(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeYHigh)
{
    auto ghostExchange = GhostExchange(subdomain);
    auto correspondingRealParticle = ghostExchange.exchangeGhosts<GhostExchange::DIRECTION_Y_HIGH>(
        particles, particles.numLocalParticles);
    EXPECT_EQ(particles.numGhostParticles, 9);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(correspondingRealParticle(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealParticle(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeYLow)
{
    auto ghostExchange = GhostExchange(subdomain);
    auto correspondingRealParticle = ghostExchange.exchangeGhosts<GhostExchange::DIRECTION_Y_LOW>(
        particles, particles.numLocalParticles);
    EXPECT_EQ(particles.numGhostParticles, 9);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(correspondingRealParticle(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealParticle(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeZHigh)
{
    auto ghostExchange = GhostExchange(subdomain);
    auto correspondingRealParticle = ghostExchange.exchangeGhosts<GhostExchange::DIRECTION_Z_HIGH>(
        particles, particles.numLocalParticles);
    EXPECT_EQ(particles.numGhostParticles, 9);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(correspondingRealParticle(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealParticle(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeZLow)
{
    auto ghostExchange = GhostExchange(subdomain);
    auto correspondingRealParticle = ghostExchange.exchangeGhosts<GhostExchange::DIRECTION_Z_LOW>(
        particles, particles.numLocalParticles);
    EXPECT_EQ(particles.numGhostParticles, 9);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(correspondingRealParticle(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealParticle(idx), particles.numLocalParticles);
        }
    }
}

TEST_F(GhostExchangeTest, SelfExchangeXYZ)
{
    auto ghostExchange = GhostExchange(subdomain);
    auto correspondingRealParticle = ghostExchange.createGhostParticlesXYZ(particles);
    EXPECT_EQ(particles.numGhostParticles, 98);
    for (auto idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        if (idx < particles.numLocalParticles)
        {
            EXPECT_EQ(correspondingRealParticle(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealParticle(idx), particles.numLocalParticles);
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
    ghostExchange.createGhostParticlesXYZ(particles);
    EXPECT_EQ(particles.numGhostParticles, 98);

    numPairs = countWithinCutoff(particles, 1.1_r, subdomain.diameter.data(), false);
    EXPECT_EQ(numPairs, 108);
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd