#include "ContributeMoleculeForceToAtoms.hpp"

#include <gtest/gtest.h>

#include "test/DiamondFixture.hpp"

namespace mrmd
{
namespace action
{
using ContributeMoleculeForceToAtomsTest = test::DiamondFixture;

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