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

#include "SystemMomentum.hpp"

namespace mrmd
{
namespace analysis
{
std::array<real_t, 3> getSystemMomentum(data::Atoms& atoms)
{
    auto vel = atoms.getVel();
    std::array<real_t, 3> velSum = {0_r, 0_r, 0_r};
    idx_t dim = 0;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    dim = 0;
    Kokkos::parallel_reduce(
        "getSystemMomentum",
        policy,
        KOKKOS_LAMBDA(const idx_t idx, real_t& sum) { sum += vel(idx, dim); },
        velSum[dim]);
    dim = 1;
    Kokkos::parallel_reduce(
        "getSystemMomentum",
        policy,
        KOKKOS_LAMBDA(const idx_t idx, real_t& sum) { sum += vel(idx, dim); },
        velSum[dim]);
    dim = 2;
    Kokkos::parallel_reduce(
        "getSystemMomentum",
        policy,
        KOKKOS_LAMBDA(const idx_t idx, real_t& sum) { sum += vel(idx, dim); },
        velSum[dim]);

    return velSum;
}

}  // namespace analysis
}  // namespace mrmd