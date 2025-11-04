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

#include "BerendsenBarostat.hpp"

namespace mrmd
{
namespace action
{
namespace BerendsenBarostat
{
void apply(data::Atoms& atoms,
           const real_t& currentPressure,
           const real_t& targetPressure,
           const real_t& gamma,
           data::Subdomain& subdomain,
           bool stretchX,
           bool stretchY,
           bool stretchZ)
{
    auto mu = std::cbrt(1_r + gamma * (currentPressure - targetPressure));
    if (stretchX) subdomain.scaleDim(mu, AXIS::X);
    if (stretchY) subdomain.scaleDim(mu, AXIS::Y);
    if (stretchZ) subdomain.scaleDim(mu, AXIS::Z);

    auto pos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        if (stretchX) pos(idx, to_underlying(AXIS::X)) *= mu;
        if (stretchY) pos(idx, to_underlying(AXIS::Y)) *= mu;
        if (stretchZ) pos(idx, to_underlying(AXIS::Z)) *= mu;
    };
    Kokkos::parallel_for("BerendsenBarostat::apply", policy, kernel);

    Kokkos::fence();
}
}  // namespace BerendsenBarostat
}  // namespace action
}  // namespace mrmd
