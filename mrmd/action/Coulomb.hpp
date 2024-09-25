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

#include "data/Atoms.hpp"
#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace action
{
namespace impl
{
class Coulomb
{
public:
    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        real_t prefac = 138.935458_r * q1 * q2;
        return prefac / distSqr;
    }

    KOKKOS_INLINE_FUNCTION
    real_t computeEnergy(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        auto r = std::sqrt(distSqr);
        real_t prefac = 138.935458_r * q1 * q2;
        return prefac / r;
    }
};
}  // namespace impl

}  // namespace action
}  // namespace mrmd