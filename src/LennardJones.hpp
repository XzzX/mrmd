#pragma once

#include "Particles.hpp"
#include "datatypes.hpp"

class LennardJones
{
private:
    real_t sigma_;
    real_t epsilon_;
    real_t sig2_;
    real_t sig6_;
    real_t ff1_;
    real_t ff2_;
    real_t rcSqr_;
    Particles::pos_t pos_;
    Particles::force_t force_;

public:
    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& dx, const real_t& dy, const real_t& dz) const
    {
        auto distSqr = dx * dx + dy * dy + dz * dz;

        if (distSqr > rcSqr_) return 0_r;

        auto frac2 = 1.0 / distSqr;
        auto frac6 = frac2 * frac2 * frac2;
        return frac6 * (ff1_ * frac6 - ff2_) * std::sqrt(frac2);
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx, const idx_t& jdx) const
    {
        auto dx = pos_(idx, 0) - pos_(jdx, 0);
        auto dy = pos_(idx, 1) - pos_(jdx, 1);
        auto dz = pos_(idx, 2) - pos_(jdx, 2);

        auto ffactor = computeForce(dx, dy, dz);

        force_(idx, 0) += dx * ffactor;
        force_(idx, 1) += dy * ffactor;
        force_(idx, 2) += dz * ffactor;

        force_(jdx, 0) -= dx * ffactor;
        force_(jdx, 1) -= dy * ffactor;
        force_(jdx, 2) -= dz * ffactor;
    }

    LennardJones(Particles& particles, const real_t rc, const real_t& sigma, const real_t& epsilon)
        : sigma_(sigma), epsilon_(epsilon)
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

class LennardJonesEnergy
{
private:
    real_t sigma_;
    real_t epsilon_;
    real_t sig2_;
    real_t sig6_;
    real_t ff1_;
    real_t ff2_;
    real_t rcSqr_;
    Particles::pos_t pos_;
    Particles::force_t force_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx, const idx_t& jdx, real_t& energy) const
    {
        auto dx = pos_(idx, 0) - pos_(jdx, 0);
        auto dy = pos_(idx, 1) - pos_(jdx, 1);
        auto dz = pos_(idx, 2) - pos_(jdx, 2);
        auto distSqr = dx * dx + dy * dy + dz * dz;

        if (distSqr > rcSqr_) return;

        real_t frac2 = sigma_ * sigma_ / distSqr;
        real_t frac6 = frac2 * frac2 * frac2;
        energy += 4.0 * epsilon_ * (frac6 * frac6 - frac6);
    }

    LennardJonesEnergy(Particles& particles,
                       const real_t rc,
                       const real_t& sigma,
                       const real_t& epsilon)
        : sigma_(sigma), epsilon_(epsilon)
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