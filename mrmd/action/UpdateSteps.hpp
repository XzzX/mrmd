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
void stepKick(real_t& vel_x,
              real_t& vel_y,
              real_t& vel_z,
              const real_t& force_x,
              const real_t& force_y,
              const real_t& force_z,
              const real_t& updateFactor)
{
    vel_x += updateFactor * force_x;
    vel_y += updateFactor * force_y;
    vel_z += updateFactor * force_z;
}

KOKKOS_INLINE_FUNCTION
real_t stepDrift(real_t& pos_x,
                 real_t& pos_y,
                 real_t& pos_z,
                 const real_t& vel_x,
                 const real_t& vel_y,
                 const real_t& vel_z,
                 const real_t& updateFactor)
{
    auto dx = updateFactor * vel_x;
    auto dy = updateFactor * vel_y;
    auto dz = updateFactor * vel_z;
    pos_x += dx;
    pos_y += dy;
    pos_z += dz;

    auto distSqr = dx * dx + dy * dy + dz * dz;
    return distSqr;
}

}  // namespace action
}  // namespace mrmd