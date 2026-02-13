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

namespace mrmd
{
namespace action
{
real_t VelocityVerletLangevinThermostat::preForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    return preForceIntegrate_apply_if(atoms, dt, KOKKOS_LAMBDA(const real_t, const real_t, const real_t) { return true; });
}

void VelocityVerletLangevinThermostat::postForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    auto dtf = 0.5_r * dt;
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        auto dtfm = dtf / mass(idx);
        vel(idx, 0) += dtfm * force(idx, 0);
        vel(idx, 1) += dtfm * force(idx, 1);
        vel(idx, 2) += dtfm * force(idx, 2);
    };
    Kokkos::parallel_for("VelocityVerletLangevinThermostat::postForceIntegrate", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd