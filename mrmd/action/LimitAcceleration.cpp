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

#include "LimitAcceleration.hpp"

#include <Kokkos_Core.hpp>

namespace mrmd::action
{
void limitAccelerationPerComponent(data::Atoms& atoms, const real_t& maxAccelerationPerComponent)
{
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        auto m = mass(idx);
        auto invM = 1_r / m;

        force(idx, 0) = Kokkos::min(force(idx, 0) * invM, +maxAccelerationPerComponent) * m;
        force(idx, 0) = Kokkos::max(force(idx, 0) * invM, -maxAccelerationPerComponent) * m;

        force(idx, 1) = Kokkos::min(force(idx, 1) * invM, +maxAccelerationPerComponent) * m;
        force(idx, 1) = Kokkos::max(force(idx, 1) * invM, -maxAccelerationPerComponent) * m;

        force(idx, 2) = Kokkos::min(force(idx, 2) * invM, +maxAccelerationPerComponent) * m;
        force(idx, 2) = Kokkos::max(force(idx, 2) * invM, -maxAccelerationPerComponent) * m;
    };
    Kokkos::parallel_for("limitForcePerComponent", policy, kernel);
    Kokkos::fence();
}

}  // namespace mrmd::action