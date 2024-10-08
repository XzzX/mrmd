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

#include "Pressure.hpp"

#include "util/math.hpp"

namespace mrmd
{
namespace analysis
{
real_t getPressure(data::Atoms& atoms, const data::Subdomain& subdomain)
{
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    real_t pressure = 0_r;
    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms + atoms.numGhostAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, real_t& sum)
    {
        sum += mass(idx) *
               (vel(idx, 0) * vel(idx, 0) + vel(idx, 1) * vel(idx, 1) + vel(idx, 2) * vel(idx, 2));

        real_t x[3];
        x[0] = pos(idx, 0);
        x[1] = pos(idx, 1);
        x[2] = pos(idx, 2);

        real_t F[3];
        F[0] = force(idx, 0);
        F[1] = force(idx, 1);
        F[2] = force(idx, 2);

        sum += util::dot3(F, x);
    };
    Kokkos::parallel_reduce("getPressure", policy, kernel, pressure);
    auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    return pressure / (3_r * volume);
}

}  // namespace analysis
}  // namespace mrmd