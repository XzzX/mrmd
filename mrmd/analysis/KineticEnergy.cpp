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

#include "KineticEnergy.hpp"

namespace mrmd
{
namespace analysis
{
real_t getKineticEnergy(data::Atoms& atoms)
{
    auto vel = atoms.getVel();
    auto mass = atoms.getMass();
    real_t velSqr = 0_r;
    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, real_t& sum)
    {
        sum += mass(idx) *
               (vel(idx, 0) * vel(idx, 0) + vel(idx, 1) * vel(idx, 1) + vel(idx, 2) * vel(idx, 2));
    };
    Kokkos::parallel_reduce("getKineticEnergy", policy, kernel, velSqr);
    return 0.5_r * velSqr;
}

real_t getMeanKineticEnergy(data::Atoms& atoms)
{
    return getKineticEnergy(atoms) / real_c(atoms.numLocalAtoms);
}

}  // namespace analysis
}  // namespace mrmd