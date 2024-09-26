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

#include "LangevinThermostat.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using LangevinThermostatTest = test::SingleAtom;

TEST_F(LangevinThermostatTest, Simple)
{
    LangevinThermostat langevinThermostat(0.5_r, 0.5_r, 0.1_r);
    langevinThermostat.apply(atoms);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);
    EXPECT_GE(force(0, 0),
              9_r + langevinThermostat.getPref1() * 1.5_r * 7_r -
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_LE(force(0, 0),
              9_r + langevinThermostat.getPref1() * 1.5_r * 7_r +
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_GE(force(0, 1),
              7_r + langevinThermostat.getPref1() * 1.5_r * 5_r -
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_LE(force(0, 1),
              7_r + langevinThermostat.getPref1() * 1.5_r * 5_r +
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_GE(force(0, 2),
              8_r + langevinThermostat.getPref1() * 1.5_r * 3_r -
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_LE(force(0, 2),
              8_r + langevinThermostat.getPref1() * 1.5_r * 3_r +
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
}
}  // namespace action
}  // namespace mrmd