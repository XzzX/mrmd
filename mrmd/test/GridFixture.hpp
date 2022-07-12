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
        for (real_t x = subdomain.minCorner[0] + real_t(0.5); x < subdomain.maxCorner[0];
             x += real_t(1))
            for (real_t y = subdomain.minCorner[1] + real_t(0.5); y < subdomain.maxCorner[1];
                 y += real_t(1))
                for (real_t z = subdomain.minCorner[2] + real_t(0.5); z < subdomain.maxCorner[2];
                     z += real_t(1))
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
        for (real_t x = subdomain.minCorner[0] + real_t(0.5); x < subdomain.maxCorner[0];
             x += real_t(1))
            for (real_t y = subdomain.minCorner[1] + real_t(0.5); y < subdomain.maxCorner[1];
                 y += real_t(1))
                for (real_t z = subdomain.minCorner[2] + real_t(0.5); z < subdomain.maxCorner[2];
                     z += real_t(1))
                {
                    for (auto i = 0; i < atomsPerMolecule; ++i)
                    {
                        atomsPos(idx, 0) = x + real_t(0.1) * real_c(i);
                        atomsPos(idx, 1) = y + real_t(0.2) * real_c(i);
                        atomsPos(idx, 2) = z + real_t(0.3) * real_c(i);
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

    data::Subdomain subdomain = data::Subdomain(
        {real_t(0), real_t(0), real_t(0)}, {real_t(3), real_t(3), real_t(3)}, real_t(0.7));
    data::Molecules molecules = data::Molecules(200);
    data::Atoms atoms = data::Atoms(200);
};

}  // namespace test
}  // namespace mrmd