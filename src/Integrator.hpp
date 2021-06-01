#pragma once

#include <Kokkos_Core.hpp>

#include "Particles.hpp"
#include "datatypes.hpp"

class Integrator
{
private:
    real_t dtf_ = 0;
    real_t dtv_ = 0;
    Particles::pos_t pos_;
    Particles::vel_t vel_;
    Particles::force_t force_;

    struct TagPreForce
    {
    };
    struct TagPostForce
    {
    };

    using PreForcePolicy =
        Kokkos::RangePolicy<Kokkos::Serial, TagPreForce, Kokkos::IndexType<idx_t>>;
    using PostForcePolicy =
        Kokkos::RangePolicy<Kokkos::Serial, TagPostForce, Kokkos::IndexType<idx_t>>;

public:
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

    Integrator(const real_t& dt) : dtf_(0.5_r * dt), dtv_(dt) {}

    void preForceIntegrate(Particles& particles)
    {
        pos_ = particles.getPos();
        vel_ = particles.getVel();
        force_ = particles.getForce();

        Kokkos::parallel_for("preForceIntegrate", PreForcePolicy(0, particles.size()), *this);
    }
    void postForceIntegrate(Particles& particles)
    {
        pos_ = particles.getPos();
        vel_ = particles.getVel();
        force_ = particles.getForce();

        Kokkos::parallel_for("postForceIntegrate", PostForcePolicy(0, particles.size()), *this);
    }
};