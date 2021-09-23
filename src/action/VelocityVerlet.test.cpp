#include "VelocityVerlet.hpp"

#include <gtest/gtest.h>

#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using VelocityVerletTest = test::SingleAtom;

TEST_F(VelocityVerletTest, preForceIntegration)
{
    auto dt = 4_r;
    VelocityVerlet::preForceIntegrate(atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), 9_r);
    EXPECT_FLOAT_EQ(force(0, 1), 7_r);
    EXPECT_FLOAT_EQ(force(0, 2), 8_r);

    EXPECT_FLOAT_EQ(vel(0, 0), 19_r);
    EXPECT_FLOAT_EQ(vel(0, 1), 14.333333_r);
    EXPECT_FLOAT_EQ(vel(0, 2), 13.666667_r);

    EXPECT_FLOAT_EQ(pos(0, 0), 78_r);
    EXPECT_FLOAT_EQ(pos(0, 1), 60.333332_r);
    EXPECT_FLOAT_EQ(pos(0, 2), 58.666668_r);
}

TEST_F(VelocityVerletTest, postForceIntegration)
{
    auto dt = 4_r;
    VelocityVerlet::postForceIntegrate(atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), 9_r);
    EXPECT_FLOAT_EQ(force(0, 1), 7_r);
    EXPECT_FLOAT_EQ(force(0, 2), 8_r);

    EXPECT_FLOAT_EQ(vel(0, 0), 19_r);
    EXPECT_FLOAT_EQ(vel(0, 1), 14.333333_r);
    EXPECT_FLOAT_EQ(vel(0, 2), 13.666667_r);

    EXPECT_FLOAT_EQ(pos(0, 0), 2_r);
    EXPECT_FLOAT_EQ(pos(0, 1), 3_r);
    EXPECT_FLOAT_EQ(pos(0, 2), 4_r);
}

}  // namespace action
}  // namespace mrmd