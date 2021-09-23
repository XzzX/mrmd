#include "VelocityScaling.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using VelocityScalingTest = test::SingleAtom;

TEST_F(VelocityScalingTest, gamma_0)
{
    constexpr real_t gamma = 0_r;
    constexpr real_t targetTemperature = 3.8_r;
    VelocityScaling velocityScaling(gamma, targetTemperature);
    velocityScaling.apply(atoms);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto mass = Cabana::slice<data::Atoms::MASS>(hAoSoA);
    auto Ekin =
        0.5_r * mass(0) * (vel(0, 0) * vel(0, 0) + vel(0, 1) * vel(0, 1) + vel(0, 2) * vel(0, 2));
    auto T = Ekin * 2_r / 3_r;
    EXPECT_FLOAT_EQ(T, 41.5_r);
}

TEST_F(VelocityScalingTest, gamma_1)
{
    constexpr real_t gamma = 1_r;
    constexpr real_t targetTemperature = 3.8_r;
    VelocityScaling velocityScaling(gamma, targetTemperature);
    velocityScaling.apply(atoms);
    velocityScaling.apply(atoms);
    velocityScaling.apply(atoms);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto mass = Cabana::slice<data::Atoms::MASS>(hAoSoA);
    auto Ekin =
        0.5_r * mass(0) * (vel(0, 0) * vel(0, 0) + vel(0, 1) * vel(0, 1) + vel(0, 2) * vel(0, 2));
    auto T = Ekin * 2_r / 3_r;
    EXPECT_FLOAT_EQ(T, targetTemperature);
}
}  // namespace action
}  // namespace mrmd