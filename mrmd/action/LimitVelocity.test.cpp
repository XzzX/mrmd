#include "LimitVelocity.hpp"

#include <gtest/gtest.h>

#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using LimitVelocityTest = test::SingleAtom;

TEST_F(LimitVelocityTest, VelocityPerComponent)
{
    limitVelocityPerComponent(atoms, real_t(0.5));

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);

    EXPECT_FLOAT_EQ(vel(0, 0), real_t(0.5));
    EXPECT_FLOAT_EQ(vel(0, 1), real_t(0.5));
    EXPECT_FLOAT_EQ(vel(0, 2), real_t(0.5));
}

}  // namespace action
}  // namespace mrmd