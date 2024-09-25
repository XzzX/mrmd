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

#include "AccumulateForce.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
namespace AccumulateForce
{
void ghostToReal(data::Atoms& atoms, const IndexView& correspondingRealAtom)
{
    data::Atoms::force_t::atomic_access_slice force = atoms.getForce();

    auto policy =
        Kokkos::RangePolicy<>(atoms.numLocalAtoms, atoms.numLocalAtoms + atoms.numGhostAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        if (correspondingRealAtom(idx) == -1) return;

        auto realIdx = correspondingRealAtom(idx);
        assert(correspondingRealAtom(realIdx) == -1 &&
               "We do not want to add forces to ghost atoms!");
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            force(realIdx, dim) += force(idx, dim);
            force(idx, dim) = 0_r;
        }
    };

    Kokkos::parallel_for("AccumulateForce::ghostToReal", policy, kernel);
    Kokkos::fence();
}
}  // namespace AccumulateForce
}  // namespace impl
}  // namespace communication
}  // namespace mrmd