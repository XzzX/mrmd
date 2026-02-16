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

#include <Kokkos_Random.hpp>

#include "action/UpdateSteps.hpp"
#include "data/Atoms.hpp"
#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace action
{
class VelocityVerletLangevinThermostat
{
private:
    Kokkos::Random_XorShift1024_Pool<> randPool_ = Kokkos::Random_XorShift1024_Pool<>(1234);
    real_t zeta_;
    real_t temperature_;

public:
    void set(const real_t& zeta, const real_t& temperature)
    {
        zeta_ = zeta;
        temperature_ = temperature;
    }

    VelocityVerletLangevinThermostat(const real_t& zeta, const real_t& temperature)
    {
        set(zeta, temperature);
    }

    real_t preForceIntegrate(data::Atoms& atoms, const real_t dt);
    void postForceIntegrate(data::Atoms& atoms, const real_t dt);

    template <std::predicate<const real_t, const real_t, const real_t> UnaryPred>
    real_t preForceIntegrate_apply_if(data::Atoms& atoms, const real_t dt, const UnaryPred& pred);
};

template <std::predicate<const real_t, const real_t, const real_t> UnaryPred>
real_t VelocityVerletLangevinThermostat::preForceIntegrate_apply_if(data::Atoms& atoms,
                                                                    const real_t dt,
                                                                    const UnaryPred& pred)
{
    auto RNG = randPool_;
    auto dtHalf(0.5_r * dt);
    auto dtFull(dt);
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    auto zeta = zeta_;
    auto temperature = temperature_;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx, real_t& maxDistSqr)
    {
        real_t dx[3];
        dx[0] = pos(idx, 0);
        dx[1] = pos(idx, 1);
        dx[2] = pos(idx, 2);

        action::updateKick(vel(idx, 0),
                           vel(idx, 1),
                           vel(idx, 2),
                           force(idx, 0),
                           force(idx, 1),
                           force(idx, 2),
                           dtHalf,
                           mass(idx));

        action::updateDrift(
            pos(idx, 0), pos(idx, 1), pos(idx, 2), vel(idx, 0), vel(idx, 1), vel(idx, 2), dtHalf);

        if (pred(pos(idx, 0), pos(idx, 1), pos(idx, 2)))
        {
            // Get a random number state from the pool for the active thread
            auto randGen = RNG.get_state();

            // Apply the Ornstein-Uhlenbeck process to the velocity
            action::updateOrnsteinUhlenbeck(vel(idx, 0),
                                            vel(idx, 1),
                                            vel(idx, 2),
                                            dtFull,
                                            mass(idx),
                                            zeta,
                                            temperature,
                                            randGen.normal(),
                                            randGen.normal(),
                                            randGen.normal());

            // Give the state back, which will allow another thread to acquire it
            RNG.free_state(randGen);
        }

        action::updateDrift(
            pos(idx, 0), pos(idx, 1), pos(idx, 2), vel(idx, 0), vel(idx, 1), vel(idx, 2), dtHalf);

        dx[0] -= pos(idx, 0);
        dx[1] -= pos(idx, 1);
        dx[2] -= pos(idx, 2);

        auto distSqr = util::dot3(dx, dx);
        maxDistSqr = Kokkos::max(distSqr, maxDistSqr);
    };
    real_t maxDistSqr = 0_r;
    Kokkos::parallel_reduce("VelocityVerletLangevinThermostat::preForceIntegrate",
                            policy,
                            kernel,
                            Kokkos::Max<real_t>(maxDistSqr));
    Kokkos::fence();
    return std::sqrt(maxDistSqr);
}

}  // namespace action
}  // namespace mrmd