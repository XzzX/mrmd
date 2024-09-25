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

#include "BerendsenThermostat.hpp"

#include <gtest/gtest.h>

#include "analysis/KineticEnergy.hpp"
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
    auto T = analysis::getMeanKineticEnergy(atoms) * (2_r / 3_r);
    action::BerendsenThermostat::apply(atoms, T, targetTemperature, gamma);
    T = analysis::getMeanKineticEnergy(atoms) * (2_r / 3_r);
    EXPECT_FLOAT_EQ(T, 41.5_r);
}

TEST_F(VelocityScalingTest, gamma_1)
{
    constexpr real_t gamma = 1_r;
    constexpr real_t targetTemperature = 3.8_r;
    auto currentTemperature = analysis::getMeanKineticEnergy(atoms) * (2_r / 3_r);
    action::BerendsenThermostat::apply(atoms, currentTemperature, targetTemperature, gamma);
    currentTemperature = analysis::getMeanKineticEnergy(atoms) * (2_r / 3_r);
    EXPECT_FLOAT_EQ(currentTemperature, targetTemperature);
}
}  // namespace action
}  // namespace mrmd