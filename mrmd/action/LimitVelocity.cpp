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

#include "LimitVelocity.hpp"

#include <Kokkos_Core.hpp>

namespace mrmd
{
namespace action
{
void limitVelocityPerComponent(data::Atoms& atoms, const real_t& maxVelocityPerComponent)
{
    auto vel = atoms.getVel();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        vel(idx, 0) = Kokkos::min(vel(idx, 0), +maxVelocityPerComponent);
        vel(idx, 0) = Kokkos::max(vel(idx, 0), -maxVelocityPerComponent);

        vel(idx, 1) = Kokkos::min(vel(idx, 1), +maxVelocityPerComponent);
        vel(idx, 1) = Kokkos::max(vel(idx, 1), -maxVelocityPerComponent);

        vel(idx, 2) = Kokkos::min(vel(idx, 2), +maxVelocityPerComponent);
        vel(idx, 2) = Kokkos::max(vel(idx, 2), -maxVelocityPerComponent);
    };
    Kokkos::parallel_for("limitVelocityPerComponent", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd