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

#pragma once

#include <Kokkos_Random.hpp>

#include "data/Atoms.hpp"
#include "datatypes.hpp"
#include "util/Random.hpp"
#include "util/ApplicationRegion.hpp"

namespace mrmd
{
namespace action
{
class LangevinThermostat
{
private:
    Kokkos::Random_XorShift1024_Pool<> randPool_ = Kokkos::Random_XorShift1024_Pool<>(1234);
    real_t pref1;
    real_t pref2;

public:
    auto getPref1() const { return pref1; }
    auto getPref2() const { return pref2; }

    void apply(data::Atoms& atoms);
    void apply(data::Atoms& atoms, const util::ApplicationRegion& applicationRegion);

    void set(const real_t gamma, const real_t temperature, const real_t timestep)
    {
        pref1 = -gamma;
        pref2 = std::sqrt(24_r * temperature * gamma / timestep);
    }

    LangevinThermostat(const real_t gamma, const real_t temperature, const real_t timestep)
    {
        set(gamma, temperature, timestep);
    }
};
}  // namespace action
}  // namespace mrmd