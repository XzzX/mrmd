#include "MultiResPeriodicGhostExchange.hpp"

#include <gtest/gtest.h>

#include "test/GridFixture.hpp"

namespace mrmd::communication::impl
{
using MultiResPeriodicGhostExchangeTest = test::GridFixture;

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsX)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostAtoms, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(200);
    ghostExchange.resetCorrespondingRealAtoms(atoms);
    ghostExchange.resetCorrespondingRealMolecules(molecules);
    auto correspondingRealAtom =
        ghostExchange.createGhostAtoms(molecules, atoms, subdomain, COORD_X);
    EXPECT_EQ(molecules.numGhostMolecules, 2 * 3 * 3);
    EXPECT_EQ(atoms.numGhostAtoms, 3 * 3 * 2 * 2);
}

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsY)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostAtoms, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(200);
    ghostExchange.resetCorrespondingRealAtoms(atoms);
    ghostExchange.resetCorrespondingRealMolecules(molecules);
    auto correspondingRealAtom =
        ghostExchange.createGhostAtoms(molecules, atoms, subdomain, COORD_Y);
    EXPECT_EQ(molecules.numGhostMolecules, 2 * 3 * 3);
    EXPECT_EQ(atoms.numGhostAtoms, 3 * 3 * 2 * 2);
}

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsZ)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostAtoms, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(200);
    ghostExchange.resetCorrespondingRealAtoms(atoms);
    ghostExchange.resetCorrespondingRealMolecules(molecules);
    auto correspondingRealAtom =
        ghostExchange.createGhostAtoms(molecules, atoms, subdomain, COORD_Z);
    EXPECT_EQ(molecules.numGhostMolecules, 2 * 3 * 3);
    EXPECT_EQ(atoms.numGhostAtoms, 3 * 3 * 2 * 2);
}

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsXYZ)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostAtoms, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(0);
    auto correspondingRealAtom = ghostExchange.createGhostAtomsXYZ(molecules, atoms, subdomain);
    EXPECT_EQ(molecules.numGhostMolecules, 5 * 5 * 5 - 3 * 3 * 3);
    EXPECT_EQ(atoms.numGhostAtoms, (5 * 5 * 5 - 3 * 3 * 3) * 2);
}

}  // namespace mrmd::communication::impl