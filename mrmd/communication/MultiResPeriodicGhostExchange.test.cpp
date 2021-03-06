#include "MultiResPeriodicGhostExchange.hpp"

#include <gtest/gtest.h>

#include "test/GridFixture.hpp"

namespace mrmd::communication::impl
{
using MultiResPeriodicGhostExchangeTest = test::GridFixture;

void multiResPeriodicGhostExchangeTestCreateGhostAtoms(data::Molecules& molecules,
                                                       data::Atoms& atoms,
                                                       data::Subdomain& subdomain,
                                                       idx_t dimension)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostAtoms, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange();
    ghostExchange.resetCorrespondingRealAtoms(atoms);
    ghostExchange.resetCorrespondingRealMolecules(molecules);
    auto correspondingRealAtom =
        ghostExchange.createGhostAtoms(molecules, atoms, subdomain, dimension);
    EXPECT_EQ(molecules.numGhostMolecules, 2 * 3 * 3);
    EXPECT_EQ(atoms.numGhostAtoms, 3 * 3 * 2 * 2);
}
TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsX)
{
    multiResPeriodicGhostExchangeTestCreateGhostAtoms(molecules, atoms, subdomain, COORD_X);
}

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsY)
{
    multiResPeriodicGhostExchangeTestCreateGhostAtoms(molecules, atoms, subdomain, COORD_Y);
}

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsZ)
{
    multiResPeriodicGhostExchangeTestCreateGhostAtoms(molecules, atoms, subdomain, COORD_Z);
}

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsXYZ)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostAtoms, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange();
    auto correspondingRealAtom = ghostExchange.createGhostAtomsXYZ(molecules, atoms, subdomain);
    EXPECT_EQ(molecules.numGhostMolecules, 5 * 5 * 5 - 3 * 3 * 3);
    EXPECT_EQ(atoms.numGhostAtoms, (5 * 5 * 5 - 3 * 3 * 3) * 2);
}

}  // namespace mrmd::communication::impl