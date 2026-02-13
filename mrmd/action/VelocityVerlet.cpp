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

#include "VelocityVerlet.hpp"

#include <algorithm>

namespace mrmd
{
namespace action
{
real_t VelocityVerlet::preForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    auto dtf(0.5_r * dt);
    auto dtv(dt);
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx, real_t& maxDistSqr)
    {
        auto dtfm = dtf / mass(idx);
        stepKick(vel(idx, 0),
                 vel(idx, 1),
                 vel(idx, 2),
                 force(idx, 0),
                 force(idx, 1),
                 force(idx, 2),
                 dtfm);

        auto distSqr = stepDrift(
            pos(idx, 0), pos(idx, 1), pos(idx, 2), vel(idx, 0), vel(idx, 1), vel(idx, 2), dtv);
        maxDistSqr = Kokkos::max(distSqr, maxDistSqr);
    };
    real_t maxDistSqr = 0_r;
    Kokkos::parallel_reduce(
        "VelocityVerlet::preForceIntegrate", policy, kernel, Kokkos::Max<real_t>(maxDistSqr));
    Kokkos::fence();
    return std::sqrt(maxDistSqr);
}

void VelocityVerlet::postForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    auto dtf = 0.5_r * dt;
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        auto dtfm = dtf / mass(idx);
        stepKick(vel(idx, 0),
                 vel(idx, 1),
                 vel(idx, 2),
                 force(idx, 0),
                 force(idx, 1),
                 force(idx, 2),
                 dtfm);
    };
    Kokkos::parallel_for("VelocityVerlet::postForceIntegrate", policy, kernel);
    Kokkos::fence();
}

KOKKOS_INLINE_FUNCTION
void VelocityVerlet::stepKick(real_t& vel_x,
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
real_t VelocityVerlet::stepDrift(real_t& pos_x,
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