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

#include <Kokkos_Core.hpp>

#include "assert/assert.hpp"

namespace mrmd
{
namespace action
{
namespace BerendsenThermostat
{
void apply(data::Atoms& atoms,
           const real_t& currentTemperature,
           const real_t& targetTemperature,
           const real_t& gamma)
{
    if (currentTemperature <= 0_r)
    {
        return;
    }

    MRMD_HOST_CHECK_GREATER(targetTemperature, 0_r);

    auto beta = std::sqrt(1_r + gamma * (targetTemperature / currentTemperature - 1_r));

    auto vel = atoms.getVel();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        vel(idx, 0) *= beta;
        vel(idx, 1) *= beta;
        vel(idx, 2) *= beta;
    };
    Kokkos::parallel_for("BerendsenThermostat::apply", policy, kernel);

    Kokkos::fence();
}
}  // namespace BerendsenThermostat
}  // namespace action
}  // namespace mrmd