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
        vel(idx, 0) += dtfm * force(idx, 0);
        vel(idx, 1) += dtfm * force(idx, 1);
        vel(idx, 2) += dtfm * force(idx, 2);
        auto dx = dtv * vel(idx, 0);
        auto dy = dtv * vel(idx, 1);
        auto dz = dtv * vel(idx, 2);
        pos(idx, 0) += dx;
        pos(idx, 1) += dy;
        pos(idx, 2) += dz;

        auto distSqr = dx * dx + dy * dy + dz * dz;
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
        vel(idx, 0) += dtfm * force(idx, 0);
        vel(idx, 1) += dtfm * force(idx, 1);
        vel(idx, 2) += dtfm * force(idx, 2);
    };
    Kokkos::parallel_for("VelocityVerlet::postForceIntegrate", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd