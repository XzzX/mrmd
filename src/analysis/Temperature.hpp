#pragma once

#include "data/Particles.hpp"

inline real_t getTemperature(Particles& particles)
{
    auto vel = particles.getVel();
    real_t velSqr = 0_r;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, particles.numLocalParticles),
        KOKKOS_LAMBDA(const idx_t idx, real_t& sum) {
            sum +=
                vel(idx, 0) * vel(idx, 0) + vel(idx, 1) * vel(idx, 1) + vel(idx, 2) * vel(idx, 2);
        },
        velSqr);

    return velSqr / (3.0 * particles.numLocalParticles);
}