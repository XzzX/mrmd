#include <gtest/gtest.h>

#include "data/Molecules.hpp"
#include "data/Atoms.hpp"

namespace mrmd
{
namespace test
{
class GridFixture : public ::testing::Test
{
protected:
    void init(const idx_t atomsPerMolecule)
    {
        molecules.resize(27 * 10);
        atoms.resize(27 * atomsPerMolecule * 10);

        auto moleculesPos = molecules.getPos();
        auto moleculesAtomsOffset = molecules.getAtomsOffset();
        auto moleculesNumAtoms = molecules.getNumAtoms();
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
                    for (auto i = 0; i < atomsPerMolecule; ++i)
                    {
                        atomsPos(idx, 0) = x + 0.1_r * real_c(i);
                        atomsPos(idx, 1) = y + 0.2_r * real_c(i);
                        atomsPos(idx, 2) = z + 0.3_r * real_c(i);
                        ++idx;
                    }
                }
        EXPECT_EQ(idx, 27 * atomsPerMolecule);
        atoms.numLocalAtoms = 27 * atomsPerMolecule;
        atoms.numGhostAtoms = 0;
        atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms);
    }

    void SetUp() override { init(2); }
    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    data::Molecules molecules = data::Molecules(200);
    data::Atoms atoms = data::Atoms(200);
};

}  // namespace test
}  // namespace mrmd