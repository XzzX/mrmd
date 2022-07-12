#pragma once

#include "data/Atoms.hpp"
#include "datatypes.hpp"
#include "util/math.hpp"

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
    real_t rc_ = real_t(0);
    real_t rcSqr_ = real_t(0);
    real_t forceShift_ = real_t(0);
    real_t energyShift_ = real_t(0);

public:
    /**
     *  DOI: 10.1140/epjst/e2016-60151-6
     *  eq 16
     */
    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        auto r = std::sqrt(distSqr);
        real_t prefac = real_t(138.935458) * q1 * q2;
        real_t expX2;
        auto erfc = util::approxErfc(alpha_ * r, expX2);
        auto force = prefac * (erfc / r + real_t(2) * alpha_ / M_SQRTPI * expX2 + r * forceShift_);
        return force / distSqr;
    }

    /**
     *  DOI: 10.1140/epjst/e2016-60151-6
     *  eq 15
     */
    KOKKOS_INLINE_FUNCTION
    real_t computeEnergy(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        auto r = std::sqrt(distSqr);
        real_t prefac = real_t(138.935458) * q1 * q2;
        auto erfc = util::approxErfc(alpha_ * r);
        return prefac * (erfc / r - energyShift_ - forceShift_ * (r - rc_));
    }

    CoulombDSF(const real_t& rc, const real_t& alpha) : alpha_(alpha), rc_(rc), rcSqr_(rc * rc)
    {
        real_t erfc = std::erfc(alpha_ * rc);
        real_t exp = std::exp(-alpha_ * alpha_ * rcSqr_);
        forceShift_ = -(erfc / rcSqr_ + real_t(2) / M_SQRTPI * alpha_ * exp / rc);
        energyShift_ = erfc / rc;
    }
};
}  // namespace impl

}  // namespace action
}  // namespace mrmd