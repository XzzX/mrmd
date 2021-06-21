#pragma once

#include <Kokkos_Core.hpp>

#include "data/Particles.hpp"
#include "datatypes.hpp"

class VelocityVerlet
{
private:
    real_t dtf_ = 0;
    real_t dtv_ = 0;
    Particles::pos_t pos_;
    Particles::vel_t vel_;
    Particles::force_t force_;

public:
    VelocityVerlet(const real_t& dt) : dtf_(0.5_r * dt), dtv_(dt) {}

    void preForceIntegrate(Particles& particles)
    {
        auto dtv = dtv_;
        auto dtf = dtf_;
        auto pos = particles.getPos();
        auto vel = particles.getVel();
        auto force = particles.getForce();

        auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
        {
            vel(idx, 0) += dtf * force(idx, 0);
            vel(idx, 1) += dtf * force(idx, 1);
            vel(idx, 2) += dtf * force(idx, 2);
            pos(idx, 0) += dtv * vel(idx, 0);
            pos(idx, 1) += dtv * vel(idx, 1);
            pos(idx, 2) += dtv * vel(idx, 2);
        };

        Kokkos::parallel_for("preForceIntegrate", particles.numLocalParticles, kernel);
    }

    void postForceIntegrate(Particles& particles)
    {
        auto dtf = dtf_;
        auto vel = particles.getVel();
        auto force = particles.getForce();

        auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
        {
            vel(idx, 0) += dtf * force(idx, 0);
            vel(idx, 1) += dtf * force(idx, 1);
            vel(idx, 2) += dtf * force(idx, 2);
        };

        Kokkos::parallel_for("postForceIntegrate", particles.numLocalParticles, kernel);
    }
};