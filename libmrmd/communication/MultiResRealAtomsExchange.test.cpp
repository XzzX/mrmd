#include "MultiResRealAtomsExchange.hpp"

#include <gtest/gtest.h>

#include "data/Subdomain.hpp"
#include "test/GridFixture.hpp"

namespace mrmd
{
namespace communication
{
using MultiResRealAtomsExchangeTest = test::GridFixture;

TEST_F(MultiResRealAtomsExchangeTest, SingleAtomTest)
{
    init(1);
    data::Subdomain domain({1_r, 1_r, 1_r}, {2_r, 2_r, 2_r}, 0.1_r);
    realAtomsExchange(domain, molecules, atoms);

    data::HostMolecules h_molecules(molecules);
    auto moleculesPos = h_molecules.getPos();
    EXPECT_GT(h_molecules.numLocalMolecules, 0);
    for (auto idx = 0; idx < h_molecules.numLocalMolecules; ++idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            EXPECT_GE(moleculesPos(idx, dim), domain.minCorner[dim]);
            EXPECT_LT(moleculesPos(idx, dim), domain.maxCorner[dim]);
        }
    }

    data::HostAtoms h_atoms(atoms);
    auto atomsPos = h_atoms.getPos();
    EXPECT_GT(h_atoms.numLocalAtoms, 0);
    for (auto idx = 0; idx < h_atoms.numLocalAtoms; ++idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            EXPECT_GE(atomsPos(idx, dim), domain.minCorner[dim]);
            EXPECT_LT(atomsPos(idx, dim), domain.maxCorner[dim]);
        }
    }
}

TEST_F(MultiResRealAtomsExchangeTest, MultiAtomTest)
{
    init(2);
    data::Subdomain domain({1_r, 1_r, 1_r}, {2_r, 2_r, 2_r}, 0.1_r);
    realAtomsExchange(domain, molecules, atoms);

    data::HostMolecules h_molecules(molecules);
    auto moleculesPos = h_molecules.getPos();
    EXPECT_GT(h_molecules.numLocalMolecules, 0);
    for (auto idx = 0; idx < h_molecules.numLocalMolecules; ++idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            EXPECT_GE(moleculesPos(idx, dim), domain.minCorner[dim]);
            EXPECT_LT(moleculesPos(idx, dim), domain.maxCorner[dim]);
        }
    }

    data::HostAtoms h_atoms(atoms);
    auto atomsPos = h_atoms.getPos();
    EXPECT_GT(h_atoms.numLocalAtoms, 0);
    for (auto idx = 0; idx < h_atoms.numLocalAtoms; ++idx)
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