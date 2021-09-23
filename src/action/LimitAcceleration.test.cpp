#include "LimitAcceleration.hpp"

#include <gtest/gtest.h>

#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using LimitAccelerationTest = test::SingleAtom;

TEST_F(LimitAccelerationTest, AccelerationPerComponent)
{
    limitAccelerationPerComponent(atoms, 0.5_r);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), 0.75_r);
    EXPECT_FLOAT_EQ(force(0, 1), 0.75_r);
    EXPECT_FLOAT_EQ(force(0, 2), 0.75_r);
}

}  // namespace action
}  // namespace mrmd