#include "MultiLayerRealParticlesExchange.hpp"

#include <gtest/gtest.h>

#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
class MultiLayerRealParticlesExchangeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto moleculesPos = molecules.getPos();
        int64_t idx = 0;
        for (real_t x = subdomain.minCorner[0] - 0.1_r; x < subdomain.maxGhostCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] - 0.1_r; y < subdomain.maxGhostCorner[1];
                 y += 1_r)
                for (real_t z = subdomain.minCorner[2] - 0.1_r; z < subdomain.maxGhostCorner[2];
                     z += 1_r)
                {
                    moleculesPos(idx, 0) = x;
                    moleculesPos(idx, 1) = y;
                    moleculesPos(idx, 2) = z;
                    ++idx;
                }
        EXPECT_EQ(idx, 8);
        molecules.numLocalParticles = idx;
        molecules.numGhostParticles = 0;
        molecules.resize(molecules.numLocalParticles + molecules.numGhostParticles);

        auto atomsPos = atoms.getPos();
        idx = 0;
        for (real_t x = subdomain.minCorner[0] - 0.1_r; x < subdomain.maxGhostCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] - 0.1_r; y < subdomain.maxGhostCorner[1];
                 y += 1_r)
                for (real_t z = subdomain.minCorner[2] - 0.1_r; z < subdomain.maxGhostCorner[2];
                     z += 1_r)
                {
                    atomsPos(idx, 0) = x;
                    atomsPos(idx, 1) = y;
                    atomsPos(idx, 2) = z;
                    ++idx;
                }
        EXPECT_EQ(idx, 8);
        atoms.numLocalParticles = idx;
        atoms.numGhostParticles = 0;
        atoms.resize(atoms.numLocalParticles + atoms.numGhostParticles);
    }

    // void TearDown() override {}

    data::Subdomain subdomain =
        data::Subdomain({0.1_r, 0.1_r, 0.1_r}, {0.9_r, 0.9_r, 0.9_r}, 0.2_r);
    data::Particles molecules = data::Particles(200);
    data::Particles atoms = data::Particles(200);
};

TEST_F(MultiLayerRealParticlesExchangeTest, SingleAtomTest)
{
    realParticlesExchange(subdomain, molecules, atoms);

    auto moleculesPos = molecules.getPos();
    for (auto idx = 0; idx < molecules.numLocalParticles; ++idx)
    {
        for (auto dim = 0; dim < data::Particles::DIMENSIONS; ++dim)
        {
            EXPECT_GE(moleculesPos(idx, dim), subdomain.minCorner[dim]);
            EXPECT_LT(moleculesPos(idx, dim), subdomain.maxCorner[dim]);
        }
    }

    auto atomsPos = atoms.getPos();
    for (auto idx = 0; idx < atoms.numLocalParticles; ++idx)
    {
        for (auto dim = 0; dim < data::Particles::DIMENSIONS; ++dim)
        {
            EXPECT_GE(atomsPos(idx, dim), subdomain.minCorner[dim]);
            EXPECT_LT(atomsPos(idx, dim), subdomain.maxCorner[dim]);
        }
    }
}

}  // namespace communication
}  // namespace mrmd