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

#include <Kokkos_Core.hpp>

#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
KOKKOS_INLINE_FUNCTION
void updateKick(real_t& vel_x,
                real_t& vel_y,
                real_t& vel_z,
                const real_t& force_x,
                const real_t& force_y,
                const real_t& force_z,
                const real_t& dt,
                const real_t& mass)

{
    auto dtfm = dt / mass;
    vel_x += dtfm * force_x;
    vel_y += dtfm * force_y;
    vel_z += dtfm * force_z;
}

KOKKOS_INLINE_FUNCTION
void updateDrift(real_t& pos_x,
                 real_t& pos_y,
                 real_t& pos_z,
                 const real_t& vel_x,
                 const real_t& vel_y,
                 const real_t& vel_z,
                 const real_t& dt)
{
    pos_x += dt * vel_x;
    pos_y += dt * vel_y;
    pos_z += dt * vel_z;
}

KOKKOS_INLINE_FUNCTION
void updateOrnsteinUhlenbeck(real_t& vel_x,
                             real_t& vel_y,
                             real_t& vel_z,
                             const real_t& dt,
                             const real_t& mass,
                             const real_t& zeta,
                             const real_t& temperature,
                             const real_t& rand_x,
                             const real_t& rand_y,
                             const real_t& rand_z)
{
    auto dtm = dt / mass;
    auto damping = std::exp(-zeta * dtm);
    auto sigma = std::sqrt(temperature / mass * (1_r - std::exp(-2_r * zeta * dtm)));

    vel_x *= damping;
    vel_y *= damping;
    vel_z *= damping;

    vel_x += sigma * rand_x;
    vel_y += sigma * rand_y;
    vel_z += sigma * rand_z;
}

}  // namespace action
}  // namespace mrmd