#include "SystemMomentum.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"

namespace mrmd
{
TEST(SystemMomentum, Simple)
{
    data::HostAtoms h_atoms(2);
    auto vel = h_atoms.getVel();
    vel(0, 0) = real_t(+2);
    vel(0, 1) = real_t(+3);
    vel(0, 2) = real_t(+4);
    vel(1, 0) = real_t(-4);
    vel(1, 1) = real_t(-8);
    vel(1, 2) = real_t(-16);
    h_atoms.numLocalAtoms = 2;
    h_atoms.numGhostAtoms = 0;
    data::Atoms atoms(h_atoms);

    auto systemMomentum = analysis::getSystemMomentum(atoms);

    EXPECT_FLOAT_EQ(systemMomentum[0], real_t(-2));
    EXPECT_FLOAT_EQ(systemMomentum[1], real_t(-5));
    EXPECT_FLOAT_EQ(systemMomentum[2], real_t(-12));
}
}  // namespace mrmd