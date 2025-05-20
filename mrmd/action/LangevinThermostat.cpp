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

#include "LangevinThermostat.hpp"

#include <Kokkos_Core.hpp>

namespace mrmd
{
namespace action
{
void LangevinThermostat::apply(data::Atoms& atoms)
{
    auto RNG = randPool_;
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    auto p1 = pref1;  // avoid capturing this pointer
    auto p2 = pref2;  // avoid capturing this pointer

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        const real_t m = mass(idx);
        const real_t mSqrt = std::sqrt(m);

        // Get a random number state from the pool for the active thread
        auto randGen = RNG.get_state();

        force(idx, 0) += p1 * vel(idx, 0) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;
        force(idx, 1) += p1 * vel(idx, 1) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;
        force(idx, 2) += p1 * vel(idx, 2) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;

        // Give the state back, which will allow another thread to acquire it
        RNG.free_state(randGen);
    };
    Kokkos::parallel_for("LangevinThermostat::applyThermostat", policy, kernel);

    Kokkos::fence();
}

void LangevinThermostat::apply(data::Atoms& atoms, const util::ApplicationRegion& applicationRegion)
    {
    auto RNG = randPool_;
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    auto p1 = pref1;  // avoid capturing this pointer
    auto p2 = pref2;  // avoid capturing this pointer

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        if (!applicationRegion.isInApplicationRegion(
            pos(idx, 0), pos(idx, 1), pos(idx, 2)))
        return;

        const real_t m = mass(idx);
        const real_t mSqrt = std::sqrt(m);

        // Get a random number state from the pool for the active thread
        auto randGen = RNG.get_state();

        force(idx, 0) += p1 * vel(idx, 0) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;
        force(idx, 1) += p1 * vel(idx, 1) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;
        force(idx, 2) += p1 * vel(idx, 2) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;

        // Give the state back, which will allow another thread to acquire it
        RNG.free_state(randGen);
    };
    Kokkos::parallel_for("LangevinThermostat::applyThermostat", policy, kernel);

    Kokkos::fence();
    }


}  // namespace action
}  // namespace mrmd