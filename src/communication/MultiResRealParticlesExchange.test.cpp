#include "MultiResRealParticlesExchange.hpp"

#include <gtest/gtest.h>

#include "data/Subdomain.hpp"
#include "test/GridFixture.hpp"

namespace mrmd
{
namespace communication
{
using MultiResRealParticlesExchangeTest = test::GridFixture;

TEST_F(MultiResRealParticlesExchangeTest, SingleAtomTest)
{
    init(1);
    data::Subdomain domain({1_r, 1_r, 1_r}, {2_r, 2_r, 2_r}, 0.1_r);
    realParticlesExchange(domain, molecules, atoms);

    auto moleculesPos = molecules.getPos();
    for (auto idx = 0; idx < molecules.numLocalMolecules; ++idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            EXPECT_GE(moleculesPos(idx, dim), domain.minCorner[dim]);
            EXPECT_LT(moleculesPos(idx, dim), domain.maxCorner[dim]);
        }
    }

    auto atomsPos = atoms.getPos();
    for (auto idx = 0; idx < atoms.numLocalParticles; ++idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            EXPECT_GE(atomsPos(idx, dim), domain.minCorner[dim]);
            EXPECT_LT(atomsPos(idx, dim), domain.maxCorner[dim]);
        }
    }
}

TEST_F(MultiResRealParticlesExchangeTest, MultiAtomTest)
{
    init(2);
    data::Subdomain domain({1_r, 1_r, 1_r}, {2_r, 2_r, 2_r}, 0.1_r);
    realParticlesExchange(domain, molecules, atoms);

    auto moleculesPos = molecules.getPos();
    for (auto idx = 0; idx < molecules.numLocalMolecules; ++idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            EXPECT_GE(moleculesPos(idx, dim), domain.minCorner[dim]);
            EXPECT_LT(moleculesPos(idx, dim), domain.maxCorner[dim]);
        }
    }

    auto atomsPos = atoms.getPos();
    for (auto idx = 0; idx < atoms.numLocalParticles; ++idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            EXPECT_GE(atomsPos(idx, dim), domain.minCorner[dim]);
            EXPECT_LT(atomsPos(idx, dim), domain.maxCorner[dim]);
        }
    }
}

}  // namespace communication
}  // namespace mrmd