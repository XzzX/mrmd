#pragma once

#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
namespace impl
{
class CoulombDSF
{
private:
    real_t alpha_;
    real_t rcSqr_ = 0_r;
    real_t forceShift_ = 0_r;
    real_t energyShift_ = 0_r;

    static constexpr real_t EWALD_F = 1.12837917;
    static constexpr real_t EWALD_P = 0.3275911;
    static constexpr real_t A1 = 0.254829592;
    static constexpr real_t A2 = -0.284496736;
    static constexpr real_t A3 = 1.421413741;
    static constexpr real_t A4 = -1.453152027;
    static constexpr real_t A5 = 1.061405429;

public:
    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        auto qqrd2e = 1_r;
        auto r = std::sqrt(distSqr);
        auto prefactor = qqrd2e * q1 * q2 / r;
        auto erfcd = std::exp(-alpha_ * alpha_ * distSqr);
        auto t = 1_r / (1_r + EWALD_P * alpha_ * r);
        auto erfcc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * erfcd;

        auto forcecoul =
            prefactor * (erfcc / r + 2_r * alpha_ / M_SQRTPI * erfcd + r * forceShift_) * r;
        auto fpair = forcecoul / distSqr;

        return fpair;
    }

    KOKKOS_INLINE_FUNCTION
    real_t computeEnergy(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        auto qqrd2e = 1_r;
        auto r = std::sqrt(distSqr);
        auto prefactor = qqrd2e * q1 * q2 / r;
        auto erfcd = std::exp(-alpha_ * alpha_ * distSqr);
        auto t = 1_r / (1_r + EWALD_P * alpha_ * r);
        auto erfcc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * erfcd;

        auto ecoul = prefactor * (erfcc - r * energyShift_ - distSqr * forceShift_);

        return ecoul;
    }

    CoulombDSF(const real_t& rc, const real_t& alpha) : alpha_(alpha), rcSqr_(rc * rc)
    {
        real_t erfcc = std::erfc(alpha_ * rc);
        real_t erfcd = std::exp(-alpha_ * alpha_ * rcSqr_);
        forceShift_ = -(erfcc / rcSqr_ + 2_r / M_SQRTPI * alpha_ * erfcd / rc);
        energyShift_ = erfcc / rc - forceShift_ * rc;
    }
};
}  // namespace impl

}  // namespace action
}  // namespace mrmd