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

namespace mrmd
{
namespace action
{
class VelocityVerletLangevinThermostat
{
private:
    Kokkos::Random_XorShift1024_Pool<> randPool_ = Kokkos::Random_XorShift1024_Pool<>(1234);
    real_t zeta_;
    real_t temperature_;

public:
    void set(const real_t& zeta, const real_t& temperature)
    {
        zeta_ = zeta;
        temperature_ = temperature;
    }

    VelocityVerletLangevinThermostat(const real_t& zeta, const real_t& temperature)
    {
        set(zeta, temperature);
    }

    real_t preForceIntegrate(data::Atoms& atoms, const real_t dt);
    std::tuple<real_t, idx_t, idx_t> preForceIntegrate(data::Atoms& atoms, const real_t dt, const real_t fluxBoundaryLeft, const real_t fluxBoundaryRight);
    void postForceIntegrate(data::Atoms& atoms, const real_t dt);
};
}  // namespace action
}  // namespace mrmd