#include "LimitVelocity.hpp"

#include <gtest/gtest.h>

#include "test/SingleParticle.hpp"

namespace mrmd
{
namespace action
{
using LimitVelocityTest = test::SingleParticle;

TEST_F(LimitVelocityTest, VelocityPerComponent)
{
    limitVelocityPerComponent(atoms, 0.5_r);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto vel = Cabana::slice<data::Particles::VEL>(hAoSoA);

    EXPECT_FLOAT_EQ(vel(0, 0), 0.5_r);
    EXPECT_FLOAT_EQ(vel(0, 1), 0.5_r);
    EXPECT_FLOAT_EQ(vel(0, 2), 0.5_r);
}

}  // namespace action
}  // namespace mrmd