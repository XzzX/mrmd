#pragma once

#include "Particles.hpp"
#include "datatypes.hpp"

class LennardJones
{
private:
    real_t sig2_;
    real_t sig6_;
    real_t ff1_;
    real_t ff2_;
    real_t rcSqr_;
    Particles::pos_t pos_;
    Particles::force_t force_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx, const idx_t& jdx) const
    {
        auto dx = pos_(idx, 0) - pos_(jdx, 0);
        auto dy = pos_(idx, 1) - pos_(jdx, 1);
        auto dz = pos_(idx, 2) - pos_(jdx, 2);
        auto distSqr = dx * dx + dy * dy + dz * dz;

        if (distSqr < rcSqr_) return;

        auto dist = std::sqrt(distSqr);
        auto frac2 = 1.0 / distSqr;
        auto frac6 = frac2 * frac2 * frac2;
        auto ffactor = frac6 * (ff1_ * frac6 - ff2_) * frac2;
        auto force = dist * ffactor;

        force_(idx, 0) += dx * force;
        force_(idx, 1) += dy * force;
        force_(idx, 2) += dz * force;

        force_(jdx, 0) -= dx * force;
        force_(jdx, 1) -= dy * force;
        force_(jdx, 2) -= dz * force;
    }

    LennardJones(Particles& particles, const real_t rc, const real_t& sigma, const real_t& epsilon)
    {
        pos_ = particles.getPos();
        force_ = particles.getForce();

        rcSqr_ = rc * rc;

        sig2_ = sigma * sigma;
        sig6_ = sig2_ * sig2_ * sig2_;
        ff1_ = 48.0 * epsilon * sig6_ * sig6_;
        ff2_ = 24.0 * epsilon * sig6_;
    }
};