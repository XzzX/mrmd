// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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