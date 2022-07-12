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
    limitAccelerationPerComponent(atoms, real_t(0.5));

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), real_t(0.75));
    EXPECT_FLOAT_EQ(force(0, 1), real_t(0.75));
    EXPECT_FLOAT_EQ(force(0, 2), real_t(0.75));
}

}  // namespace action
}  // namespace mrmd