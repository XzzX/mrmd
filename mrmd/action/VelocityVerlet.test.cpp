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
    auto dt = real_t(4);
    VelocityVerlet::preForceIntegrate(atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), real_t(9));
    EXPECT_FLOAT_EQ(force(0, 1), real_t(7));
    EXPECT_FLOAT_EQ(force(0, 2), real_t(8));

    EXPECT_FLOAT_EQ(vel(0, 0), real_t(19));
    EXPECT_FLOAT_EQ(vel(0, 1), real_t(14.333333));
    EXPECT_FLOAT_EQ(vel(0, 2), real_t(13.666667));

    EXPECT_FLOAT_EQ(pos(0, 0), real_t(78));
    EXPECT_FLOAT_EQ(pos(0, 1), real_t(60.333332));
    EXPECT_FLOAT_EQ(pos(0, 2), real_t(58.666668));
}

TEST_F(VelocityVerletTest, postForceIntegration)
{
    auto dt = real_t(4);
    VelocityVerlet::postForceIntegrate(atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), real_t(9));
    EXPECT_FLOAT_EQ(force(0, 1), real_t(7));
    EXPECT_FLOAT_EQ(force(0, 2), real_t(8));

    EXPECT_FLOAT_EQ(vel(0, 0), real_t(19));
    EXPECT_FLOAT_EQ(vel(0, 1), real_t(14.333333));
    EXPECT_FLOAT_EQ(vel(0, 2), real_t(13.666667));

    EXPECT_FLOAT_EQ(pos(0, 0), real_t(2));
    EXPECT_FLOAT_EQ(pos(0, 1), real_t(3));
    EXPECT_FLOAT_EQ(pos(0, 2), real_t(4));
}

}  // namespace action
}  // namespace mrmd