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

#include "LennardJones.hpp"

#include <algorithm>

namespace mrmd::action
{
void LennardJones::apply(data::Atoms& atoms, HalfVerletList& verletList)
{
    energyAndVirial_ = data::EnergyAndVirialReducer();

    pos_ = atoms.getPos();
    force_ = atoms.getForce();
    type_ = atoms.getType();
    verletList_ = verletList;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    Kokkos::parallel_reduce("LennardJones::applyForces", policy, *this, energyAndVirial_);
    Kokkos::fence();
}

LennardJones::LennardJones(const real_t rc,
                           const real_t& sigma,
                           const real_t& epsilon,
                           const real_t& cappingDistance)
    : LennardJones({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, false)
{
}

LennardJones::LennardJones(const std::vector<real_t>& cappingDistance,
                           const std::vector<real_t>& rc,
                           const std::vector<real_t>& sigma,
                           const std::vector<real_t>& epsilon,
                           const idx_t& numTypes,
                           const bool isShifted)
    : LJ_(cappingDistance, rc, sigma, epsilon, numTypes, isShifted), numTypes_(1)
{
    auto rcMax = std::ranges::max(rc);
    rcSqr_ = rcMax * rcMax;
}
}  // namespace mrmd::action