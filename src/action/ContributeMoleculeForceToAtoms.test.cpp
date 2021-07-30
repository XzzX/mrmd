#include "ContributeMoleculeForceToAtoms.hpp"

#include <gtest/gtest.h>

#include "data/Molecules.hpp"
#include "data/Particles.hpp"

namespace mrmd
{
namespace action
{
/**
 * 2 molecules with 2 atoms each
 * +++++
 * ++A++
 * +B+A+
 * ++B++
 * +++++
 */
class ContributeMoleculeForceToAtomsTest : public ::testing::Test
{
protected:
    static auto getMolecules()
    {
        data::Molecules molecules(2);

        auto moleculesAtomsEndIdx = molecules.getAtomsEndIdx();
        moleculesAtomsEndIdx(0) = 2;
        moleculesAtomsEndIdx(1) = 4;

        molecules.numLocalMolecules = 2;

        return molecules;
    }

    static auto getAtoms()
    {
        data::Particles atoms(4);

        auto pos = atoms.getPos();
        pos(0, 0) = 1_r;
        pos(0, 1) = 0_r;
        pos(0, 2) = 0_r;
        atoms.getRelativeMass()(0) = 0.5_r;

        pos(1, 0) = 0_r;
        pos(1, 1) = 1_r;
        pos(1, 2) = 0_r;
        atoms.getRelativeMass()(1) = 0.5_r;

        pos(2, 0) = -1_r;
        pos(2, 1) = 0_r;
        pos(2, 2) = 0_r;
        atoms.getRelativeMass()(2) = 0.5_r;

        pos(3, 0) = 0_r;
        pos(3, 1) = -1_r;
        pos(3, 2) = 0_r;
        atoms.getRelativeMass()(3) = 0.5_r;

        atoms.numLocalParticles = 4;

        return atoms;
    }

    void SetUp() override
    {
        molecules = getMolecules();
        atoms = getAtoms();
    }

    // void TearDown() override {}

    data::Molecules molecules = data::Molecules(1);
    data::Particles atoms = data::Particles(1);
};

TEST_F(ContributeMoleculeForceToAtomsTest, update)
{
    molecules.getForce()(0, 0) = 5_r;
    molecules.getForce()(0, 1) = 6_r;
    molecules.getForce()(0, 2) = 7_r;

    action::ContributeMoleculeForceToAtoms::update(molecules, atoms);

    EXPECT_FLOAT_EQ(atoms.getForce()(0, 0), 2.5_r);
    EXPECT_FLOAT_EQ(atoms.getForce()(0, 1), 3.0_r);
    EXPECT_FLOAT_EQ(atoms.getForce()(0, 2), 3.5_r);
    EXPECT_FLOAT_EQ(atoms.getForce()(1, 0), 2.5_r);
    EXPECT_FLOAT_EQ(atoms.getForce()(1, 1), 3.0_r);
    EXPECT_FLOAT_EQ(atoms.getForce()(1, 2), 3.5_r);
}

}  // namespace action
}  // namespace mrmd