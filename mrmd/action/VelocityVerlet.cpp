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

#include "action/UpdateSteps.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace action
{
real_t VelocityVerlet::preForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    auto dtHalf(0.5_r * dt);
    auto dtFull(dt);
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx, real_t& maxDistSqr)
    {
        real_t dx[3];
        dx[0] = pos(idx, 0);
        dx[1] = pos(idx, 1);
        dx[2] = pos(idx, 2);

        action::updateKick(vel(idx, 0),
                           vel(idx, 1),
                           vel(idx, 2),
                           force(idx, 0),
                           force(idx, 1),
                           force(idx, 2),
                           dtHalf,
                           mass(idx));

        action::updateDrift(
            pos(idx, 0), pos(idx, 1), pos(idx, 2), vel(idx, 0), vel(idx, 1), vel(idx, 2), dtFull);

        dx[0] -= pos(idx, 0);
        dx[1] -= pos(idx, 1);
        dx[2] -= pos(idx, 2);

        auto distSqr = util::dot3(dx, dx);
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
    auto dtHalf(0.5_r * dt);
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        action::updateKick(vel(idx, 0),
                           vel(idx, 1),
                           vel(idx, 2),
                           force(idx, 0),
                           force(idx, 1),
                           force(idx, 2),
                           dtHalf,
                           mass(idx));
    };
    Kokkos::parallel_for("VelocityVerlet::postForceIntegrate", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd