#include "MultiLayerPeriodicGhostExchange.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class MultiLayerPeriodicGhostExchangeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto moleculesPos = molecules.getPos();
        int64_t idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    moleculesPos(idx, 0) = x;
                    moleculesPos(idx, 1) = y;
                    moleculesPos(idx, 2) = z;
                    ++idx;
                }
        EXPECT_EQ(idx, 27);
        molecules.numLocalParticles = 27;
        molecules.numGhostParticles = 0;
        molecules.resize(molecules.numLocalParticles + molecules.numGhostParticles);

        auto atomsPos = atoms.getPos();
        idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    atomsPos(idx, 0) = x;
                    atomsPos(idx, 1) = y;
                    atomsPos(idx, 2) = z;
                    ++idx;
                }
        EXPECT_EQ(idx, 27);
        atoms.numLocalParticles = 27;
        atoms.numGhostParticles = 0;
        atoms.resize(atoms.numLocalParticles + atoms.numGhostParticles);
    }

    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    data::Particles molecules = data::Particles(200);
    data::Particles atoms = data::Particles(200);
};

TEST_F(MultiLayerPeriodicGhostExchangeTest, SelfExchangeXHigh)
{
    EXPECT_EQ(molecules.numGhostParticles, 0);
    EXPECT_EQ(atoms.numGhostParticles, 0);
    auto ghostExchange = MultiLayerPeriodicGhostExchange(subdomain);
    auto correspondingRealParticle =
        ghostExchange.exchangeGhosts<MultiLayerPeriodicGhostExchange::DIRECTION_X_HIGH>(
            molecules, atoms, molecules.numLocalParticles);
    EXPECT_EQ(molecules.numGhostParticles, 9);
    EXPECT_EQ(atoms.numGhostParticles, 9);
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd