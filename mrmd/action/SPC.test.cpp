#include "SPC.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "util/angle.hpp"

namespace mrmd
{
namespace action
{
class SPC_Test : public ::testing::Test
{
protected:
    static auto getMolecules()
    {
        data::HostMolecules molecules(2);

        auto pos = molecules.getPos();
        pos(0, 0) = 0_r;
        pos(0, 1) = 0_r;
        pos(0, 2) = 0_r;

        auto moleculesAtomsOffset = molecules.getAtomsOffset();
        auto moleculesNumAtoms = molecules.getNumAtoms();
        moleculesAtomsOffset(0) = 0;
        moleculesNumAtoms(0) = 3;

        molecules.numLocalMolecules = 1;
        molecules.numGhostMolecules = 0;

        return molecules;
    }

    static auto getAtoms()
    {
        data::HostAtoms atoms(6);

        auto pos = atoms.getPos();
        auto vel = atoms.getVel();
        auto charge = atoms.getCharge();
        auto type = atoms.getType();
        auto mass = atoms.getMass();
        auto realtiveMass = atoms.getRelativeMass();

        idx_t idx = 0;
        pos(idx * 3 + 0, 0) = 0_r;
        pos(idx * 3 + 0, 1) = 0_r;
        pos(idx * 3 + 0, 2) = 0_r;

        type(idx * 3 + 0) = 0;
        mass(idx * 3 + 0) = 15.999_r;
        charge(idx * 3 + 0) = -0.82_r;
        realtiveMass(idx * 3 + 0) = 15.999_r / (15.999_r + 2_r * 1.008_r);

        // hydrogen 1
        pos(idx * 3 + 1, 0) = pos(idx * 3 + 0, 0) + SPC::eqDistanceHO;
        pos(idx * 3 + 1, 1) = pos(idx * 3 + 0, 1);
        pos(idx * 3 + 1, 2) = pos(idx * 3 + 0, 2);

        //        vel(idx * 3 + 1, 0) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 1, 1) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 1, 2) = (randGen.drand() - 0.5_r) * 1_r;

        type(idx * 3 + 1) = 1;
        mass(idx * 3 + 1) = 1.008_r;
        charge(idx * 3 + 1) = +0.41_r;
        realtiveMass(idx * 3 + 1) = 1.008_r / (15.999_r + 2_r * 1.008_r);

        // hydrogen 2
        pos(idx * 3 + 2, 0) = pos(idx * 3 + 0, 0) + SPC::eqDistanceHO * std::cos(SPC::angleHOH);
        pos(idx * 3 + 2, 1) = pos(idx * 3 + 0, 1) + SPC::eqDistanceHO * std::sin(SPC::angleHOH);
        pos(idx * 3 + 2, 2) = pos(idx * 3 + 0, 2);

        //        vel(idx * 3 + 2, 0) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 2, 1) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 2, 2) = (randGen.drand() - 0.5_r) * 1_r;

        type(idx * 3 + 2) = 1;
        mass(idx * 3 + 2) = 1.008_r;
        charge(idx * 3 + 2) = +0.41_r;
        realtiveMass(idx * 3 + 2) = 1.008_r / (15.999_r + 2_r * 1.008_r);

        atoms.numLocalAtoms = 3;
        atoms.numGhostAtoms = 0;

        return atoms;
    }

    void SetUp() override
    {
        data::deep_copy(molecules, getMolecules());
        data::deep_copy(atoms, getAtoms());

        auto cutoff = 2_r;
        auto cellRatio = 1_r;
        real_t minGrid[3] = {-1_r, -1_r, -1_r};
        real_t maxGrid[3] = {+1_r, +1_r, +1_r};
        auto expectedNumNeighbors = 4;
        moleculesVerletList.build(molecules.getPos(),
                                  0,
                                  molecules.numLocalMolecules,
                                  cutoff,
                                  cellRatio,
                                  minGrid,
                                  maxGrid,
                                  expectedNumNeighbors);
    }

    // void TearDown() override {}

    data::Molecules molecules = data::Molecules(1);
    data::Atoms atoms = data::Atoms(1);
    HalfVerletList moleculesVerletList;
};

TEST_F(SPC_Test, CHECK_CONSTRAINTS)
{
    constexpr auto dt = 0.1_r;

    SPC spc;
    spc.enforcePositionalConstraints(molecules, atoms, dt);

    data::HostAtoms h_atoms(atoms);
    auto force = h_atoms.getForce();
    EXPECT_FLOAT_EQ(force(0, 0) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(force(0, 1) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(force(0, 2) + 1_r, 1_r);

    EXPECT_FLOAT_EQ(force(1, 0) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(force(1, 1) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(force(1, 2) + 1_r, 1_r);

    EXPECT_FLOAT_EQ(force(2, 0) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(force(2, 1) + 1_r, 1_r);
    EXPECT_FLOAT_EQ(force(2, 2) + 1_r, 1_r);
}

}  // namespace action
}  // namespace mrmd