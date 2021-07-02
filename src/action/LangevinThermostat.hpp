#pragma once

#include <Kokkos_Random.hpp>

#include "data/Particles.hpp"
#include "datatypes.hpp"
#include "util/Random.hpp"

class LangevinThermostat
{
private:
    data::Particles::vel_t vel_;
    data::Particles::force_t force_;

    util::Random rng;

public:
    const real_t pref1;
    const real_t pref2;

    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        real_t mass = 1_r;
        real_t massSqrt = 1_r;  // std::sqrt(mass);

        force_(idx, 0) += pref1 * vel_(idx, 0) * mass + pref2 * (rng.draw() - 0.5_r) * massSqrt;
        force_(idx, 1) += pref1 * vel_(idx, 1) * mass + pref2 * (rng.draw() - 0.5_r) * massSqrt;
        force_(idx, 2) += pref1 * vel_(idx, 2) * mass + pref2 * (rng.draw() - 0.5_r) * massSqrt;
    }

    void applyThermostat(data::Particles& particles)
    {
        vel_ = particles.getVel();
        force_ = particles.getForce();

        auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
        Kokkos::parallel_for(policy, *this, "LangevinThermostat::applyThermostat");

        Kokkos::fence();
    }

    LangevinThermostat(const real_t gamma, const real_t temperature, const real_t timestep)
        : pref1(-gamma), pref2(std::sqrt(24_r * temperature * gamma / timestep))
    {
    }
};