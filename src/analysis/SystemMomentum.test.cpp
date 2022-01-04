#include "SystemMomentum.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"

namespace mrmd
{
TEST(SystemMomentum, Simple)
{
    data::HostAtoms h_atoms(2);
    auto vel = h_atoms.getVel();
    vel(0, 0) = +2_r;
    vel(0, 1) = +3_r;
    vel(0, 2) = +4_r;
    vel(1, 0) = -4_r;
    vel(1, 1) = -8_r;
    vel(1, 2) = -16_r;
    h_atoms.numLocalAtoms = 2;
    h_atoms.numGhostAtoms = 0;
    data::Atoms atoms(h_atoms);

    auto systemMomentum = analysis::getSystemMomentum(atoms);

    EXPECT_FLOAT_EQ(systemMomentum[0], -2_r);
    EXPECT_FLOAT_EQ(systemMomentum[1], -5_r);
    EXPECT_FLOAT_EQ(systemMomentum[2], -12_r);
}
}  // namespace mrmd