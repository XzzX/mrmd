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

#include <algorithm>

#include "VelocityVerlet.hpp"

namespace mrmd
{
namespace action
{
real_t VelocityVerletLangevinThermostat::preForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    return preForceIntegrate_apply_if(
        atoms, dt, KOKKOS_LAMBDA(const real_t, const real_t, const real_t) { return true; });
}

void VelocityVerletLangevinThermostat::postForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    action::VelocityVerlet::postForceIntegrate(atoms, dt);
}

}  // namespace action
}  // namespace mrmd