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

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"

namespace mrmd
{
namespace test
{
class GridFixture : public ::testing::Test
{
protected:
    void init(const idx_t atomsPerMolecule)
    {
        auto h_molecules = data::HostMolecules(200);
        auto h_atoms = data::HostAtoms(200);

        h_molecules.resize(27 * 10);
        h_atoms.resize(27 * atomsPerMolecule * 10);

        auto moleculesPos = h_molecules.getPos();
        auto moleculesAtomsOffset = h_molecules.getAtomsOffset();
        auto moleculesNumAtoms = h_molecules.getNumAtoms();
        int64_t idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    moleculesPos(idx, 0) = x;
                    moleculesPos(idx, 1) = y;
                    moleculesPos(idx, 2) = z;
                    moleculesAtomsOffset(idx) = idx * atomsPerMolecule;
                    moleculesNumAtoms(idx) = atomsPerMolecule;
                    ++idx;
                }
        EXPECT_EQ(idx, 27);
        h_molecules.numLocalMolecules = 27;
        h_molecules.numGhostMolecules = 0;
        h_molecules.resize(h_molecules.numLocalMolecules + h_molecules.numGhostMolecules);

        auto atomsPos = h_atoms.getPos();
        idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    for (auto i = 0; i < atomsPerMolecule; ++i)
                    {
                        atomsPos(idx, 0) = x + 0.1_r * real_c(i);
                        atomsPos(idx, 1) = y + 0.2_r * real_c(i);
                        atomsPos(idx, 2) = z + 0.3_r * real_c(i);
                        ++idx;
                    }
                }
        EXPECT_EQ(idx, 27 * atomsPerMolecule);
        h_atoms.numLocalAtoms = 27 * atomsPerMolecule;
        h_atoms.numGhostAtoms = 0;
        h_atoms.resize(h_atoms.numLocalAtoms + h_atoms.numGhostAtoms);

        data::deep_copy(molecules, h_molecules);
        data::deep_copy(atoms, h_atoms);
    }

    void SetUp() override { init(2); }
    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    data::Molecules molecules = data::Molecules(200);
    data::Atoms atoms = data::Atoms(200);
};

}  // namespace test
}  // namespace mrmd