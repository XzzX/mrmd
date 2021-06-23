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
    struct TagPreForce
    {
    };
    struct TagPostForce
    {
    };

    VelocityVerlet(const real_t& dt) : dtf_(0.5_r * dt), dtv_(dt) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(TagPreForce, const idx_t& idx) const
    {
        vel_(idx, 0) += dtf_ * force_(idx, 0);
        vel_(idx, 1) += dtf_ * force_(idx, 1);
        vel_(idx, 2) += dtf_ * force_(idx, 2);
        pos_(idx, 0) += dtv_ * vel_(idx, 0);
        pos_(idx, 1) += dtv_ * vel_(idx, 1);
        pos_(idx, 2) += dtv_ * vel_(idx, 2);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TagPostForce, const idx_t& idx) const
    {
        vel_(idx, 0) += dtf_ * force_(idx, 0);
        vel_(idx, 1) += dtf_ * force_(idx, 1);
        vel_(idx, 2) += dtf_ * force_(idx, 2);
    }

    void preForceIntegrate(Particles& particles)
    {
        pos_ = particles.getPos();
        vel_ = particles.getVel();
        force_ = particles.getForce();

        auto policy = Kokkos::RangePolicy<TagPreForce>(0, particles.numLocalParticles);
        Kokkos::parallel_for("preForceIntegrate", policy, *this);
    }

    void postForceIntegrate(Particles& particles)
    {
        vel_ = particles.getVel();
        force_ = particles.getForce();

        auto policy = Kokkos::RangePolicy<TagPostForce>(0, particles.numLocalParticles);
        Kokkos::parallel_for("postForceIntegrate", policy, *this);
    }
};