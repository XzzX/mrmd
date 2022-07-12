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
        pos(0, 0) = real_t(0);
        pos(0, 1) = real_t(0);
        pos(0, 2) = real_t(0);

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
        pos(idx * 3 + 0, 0) = real_t(0);
        pos(idx * 3 + 0, 1) = real_t(0);
        pos(idx * 3 + 0, 2) = real_t(0);

        type(idx * 3 + 0) = 0;
        mass(idx * 3 + 0) = real_t(15.999);
        charge(idx * 3 + 0) = real_t(-0.82);
        realtiveMass(idx * 3 + 0) = real_t(15.999) / (real_t(15.999) + real_t(2) * real_t(1.008));

        // hydrogen 1
        pos(idx * 3 + 1, 0) = pos(idx * 3 + 0, 0) + SPC::eqDistanceHO;
        pos(idx * 3 + 1, 1) = pos(idx * 3 + 0, 1);
        pos(idx * 3 + 1, 2) = pos(idx * 3 + 0, 2);

        //        vel(idx * 3 + 1, 0) = (randGen.drand() - real_t(0.5)) * real_t(1);
        //        vel(idx * 3 + 1, 1) = (randGen.drand() - real_t(0.5)) * real_t(1);
        //        vel(idx * 3 + 1, 2) = (randGen.drand() - real_t(0.5)) * real_t(1);

        type(idx * 3 + 1) = 1;
        mass(idx * 3 + 1) = real_t(1.008);
        charge(idx * 3 + 1) = real_t(+0.41);
        realtiveMass(idx * 3 + 1) = real_t(1.008) / (real_t(15.999) + real_t(2) * real_t(1.008));

        // hydrogen 2
        pos(idx * 3 + 2, 0) = pos(idx * 3 + 0, 0) + SPC::eqDistanceHO * std::cos(SPC::angleHOH);
        pos(idx * 3 + 2, 1) = pos(idx * 3 + 0, 1) + SPC::eqDistanceHO * std::sin(SPC::angleHOH);
        pos(idx * 3 + 2, 2) = pos(idx * 3 + 0, 2);

        //        vel(idx * 3 + 2, 0) = (randGen.drand() - real_t(0.5)) * real_t(1);
        //        vel(idx * 3 + 2, 1) = (randGen.drand() - real_t(0.5)) * real_t(1);
        //        vel(idx * 3 + 2, 2) = (randGen.drand() - real_t(0.5)) * real_t(1);

        type(idx * 3 + 2) = 1;
        mass(idx * 3 + 2) = real_t(1.008);
        charge(idx * 3 + 2) = real_t(+0.41);
        realtiveMass(idx * 3 + 2) = real_t(1.008) / (real_t(15.999) + real_t(2) * real_t(1.008));

        atoms.numLocalAtoms = 3;
        atoms.numGhostAtoms = 0;

        return atoms;
    }

    void SetUp() override
    {
        data::deep_copy(molecules, getMolecules());
        data::deep_copy(atoms, getAtoms());

        auto cutoff = real_t(2);
        auto cellRatio = real_t(1);
        real_t minGrid[3] = {real_t(-1), real_t(-1), real_t(-1)};
        real_t maxGrid[3] = {real_t(+1), real_t(+1), real_t(+1)};
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
    constexpr auto dt = real_t(0.1);

    SPC spc;
    spc.enforcePositionalConstraints(molecules, atoms, dt);

    data::HostAtoms h_atoms(atoms);
    auto force = h_atoms.getForce();
    EXPECT_FLOAT_EQ(force(0, 0) + real_t(1), real_t(1));
    EXPECT_FLOAT_EQ(force(0, 1) + real_t(1), real_t(1));
    EXPECT_FLOAT_EQ(force(0, 2) + real_t(1), real_t(1));

    EXPECT_FLOAT_EQ(force(1, 0) + real_t(1), real_t(1));
    EXPECT_FLOAT_EQ(force(1, 1) + real_t(1), real_t(1));
    EXPECT_FLOAT_EQ(force(1, 2) + real_t(1), real_t(1));

    EXPECT_FLOAT_EQ(force(2, 0) + real_t(1), real_t(1));
    EXPECT_FLOAT_EQ(force(2, 1) + real_t(1), real_t(1));
    EXPECT_FLOAT_EQ(force(2, 2) + real_t(1), real_t(1));
}

}  // namespace action
}  // namespace mrmd