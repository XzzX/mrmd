#include "MultiResPeriodicGhostExchange.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class MultiResPeriodicGhostExchangeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto moleculesPos = molecules.getPos();
        auto moleculesAtomsEndIdx = molecules.getAtomsEndIdx();
        int64_t idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    moleculesPos(idx, 0) = x;
                    moleculesPos(idx, 1) = y;
                    moleculesPos(idx, 2) = z;
                    moleculesAtomsEndIdx(idx) = (idx + 1) * 2;
                    ++idx;
                }
        EXPECT_EQ(idx, 27);
        molecules.numLocalMolecules = 27;
        molecules.numGhostMolecules = 0;
        molecules.resize(molecules.numLocalMolecules + molecules.numGhostMolecules);

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
                    atomsPos(idx, 0) = x;
                    atomsPos(idx, 1) = y;
                    atomsPos(idx, 2) = z;
                    ++idx;
                }
        EXPECT_EQ(idx, 54);
        atoms.numLocalParticles = 54;
        atoms.numGhostParticles = 0;
        atoms.resize(atoms.numLocalParticles + atoms.numGhostParticles);
    }

    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    data::Molecules molecules = data::Molecules(200);
    data::Particles atoms = data::Particles(200);
};

TEST_F(MultiResPeriodicGhostExchangeTest, SelfExchangeXHigh)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostParticles, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(subdomain);
    auto correspondingRealParticle =
        ghostExchange.exchangeGhosts<MultiResPeriodicGhostExchange::DIRECTION_X_HIGH>(
            molecules, atoms, molecules.numLocalMolecules);
    EXPECT_EQ(molecules.numGhostMolecules, 9);
    EXPECT_EQ(atoms.numGhostParticles, 9);
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd