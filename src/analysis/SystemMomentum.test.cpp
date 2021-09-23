#include "SystemMomentum.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"

namespace mrmd
{
TEST(SystemMomentum, Simple)
{
    data::Atoms atoms(2);
    auto vel = atoms.getVel();
    vel(0, 0) = +2_r;
    vel(0, 1) = +3_r;
    vel(0, 2) = +4_r;
    vel(1, 0) = -4_r;
    vel(1, 1) = -8_r;
    vel(1, 2) = -16_r;
    atoms.numLocalAtoms = 2;
    atoms.numGhostAtoms = 0;

    auto systemMomentum = analysis::getSystemMomentum(atoms);

    EXPECT_FLOAT_EQ(systemMomentum[0], -2_r);
    EXPECT_FLOAT_EQ(systemMomentum[1], -5_r);
    EXPECT_FLOAT_EQ(systemMomentum[2], -12_r);
}
}  // namespace mrmd