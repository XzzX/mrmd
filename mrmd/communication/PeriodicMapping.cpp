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

#include "PeriodicMapping.hpp"

#include <algorithm>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
namespace PeriodicMapping
{
void mapIntoDomain(data::Atoms& atoms, const data::Subdomain& subdomain)
{
    auto pos = atoms.getPos();
    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            auto& x = pos(idx, dim);
            if (subdomain.maxCorner[dim] <= x)
            {
                x -= subdomain.diameter[dim];
                x = std::max(x, subdomain.minCorner[dim]);
            }
            if (x < subdomain.minCorner[dim])
            {
                x += subdomain.diameter[dim];
                if (subdomain.maxCorner[dim] <= x)
                {
                    x = subdomain.minCorner[dim];
                }
            }
            assert(x < subdomain.maxCorner[dim]);
            assert(subdomain.minCorner[dim] <= x);
        }
    };
    Kokkos::parallel_for("PeriodicMapping::mapIntoDomain", policy, kernel);
    Kokkos::fence();
}
}  // namespace PeriodicMapping
}  // namespace impl
}  // namespace communication
}  // namespace mrmd