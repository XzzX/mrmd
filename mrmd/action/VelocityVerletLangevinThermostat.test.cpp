// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "VelocityVerletLangevinThermostat.hpp"

#include <gtest/gtest.h>

#include "assert/verbose.hpp"
#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using LangevinThermostatTest = test::SingleAtom;

struct NoThermostatPred
{
    KOKKOS_FUNCTION bool operator()(const real_t x, const real_t, const real_t) const
    {
        return false;
    }
};

struct LocalThermostatPred
{
    KOKKOS_FUNCTION bool operator()(const real_t x, const real_t, const real_t) const
    {
        return (x > 78_r);
    }
};

TEST_F(LangevinThermostatTest, Simple)
{
    VelocityVerletLangevinThermostat langevinIntegrator(0.5_r, 0.5_r);
    langevinIntegrator.preForceIntegrate(atoms, 4_r);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), 9_r);
    EXPECT_FLOAT_EQ(force(0, 1), 7_r);
    EXPECT_FLOAT_EQ(force(0, 2), 8_r);

    const real_t epsilon = 1e-6_r;
    EXPECT_FALSE(assumption::isFloatEqual(vel(0, 0), 19_r, epsilon) &&
                 assumption::isFloatEqual(vel(0, 1), 14.333333_r, epsilon) &&
                 assumption::isFloatEqual(vel(0, 2), 13.666667_r, epsilon));
    EXPECT_FALSE(assumption::isFloatEqual(pos(0, 0), 78_r, epsilon) &&
                 assumption::isFloatEqual(pos(0, 1), 60.333332_r, epsilon) &&
                 assumption::isFloatEqual(pos(0, 2), 58.666668_r, epsilon));
}

TEST_F(LangevinThermostatTest, NoThermostat)
{
    VelocityVerletLangevinThermostat langevinIntegrator(0.5_r, 0.5_r);
    langevinIntegrator.preForceIntegrate_apply_if(atoms, 4_r, NoThermostatPred());

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

TEST_F(LangevinThermostatTest, LocalThermostat)
{
    VelocityVerletLangevinThermostat langevinIntegrator(0.5_r, 0.5_r);
    langevinIntegrator.preForceIntegrate_apply_if(atoms, 4_r, LocalThermostatPred());

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

    langevinIntegrator.preForceIntegrate_apply_if(atoms, 4_r, LocalThermostatPred());

    const real_t epsilon = 1e-6_r;
    EXPECT_FALSE(assumption::isFloatEqual(vel(0, 0), 31_r, epsilon) &&
                 assumption::isFloatEqual(vel(0, 1), 23.666667_r, epsilon) &&
                 assumption::isFloatEqual(vel(0, 2), 24.333333_r, epsilon));
    EXPECT_FALSE(assumption::isFloatEqual(pos(0, 0), 202_r, epsilon) &&
                 assumption::isFloatEqual(pos(0, 1), 155_r, epsilon) &&
                 assumption::isFloatEqual(pos(0, 2), 156_r, epsilon));
}

}  // namespace action
}  // namespace mrmd