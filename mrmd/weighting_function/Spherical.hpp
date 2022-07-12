#pragma once

#include "constants.hpp"
#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace weighting_function
{
class Spherical
{
private:
    const std::array<real_t, 3> center_;
    const real_t atomisticRadius_;
    const real_t atomisticRadiusSqr_;
    const real_t hybridRegionDiameter_;
    const real_t coarseRadiusSqr_;
    const int exponent_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const real_t x,
                    const real_t y,
                    const real_t z,
                    real_t& lambda,
                    real_t& gradLambdaX,
                    real_t& gradLambdaY,
                    real_t& gradLambdaZ) const
    {
        real_t dx[3] = {x - center_[0], y - center_[1], z - center_[2]};
        auto dxSqr = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

        if (dxSqr < atomisticRadiusSqr_)
        {
            lambda = real_t(1);
            gradLambdaX = real_t(0);
            gradLambdaY = real_t(0);
            gradLambdaZ = real_t(0);
            return;
        }
        if (dxSqr > coarseRadiusSqr_)
        {
            lambda = real_t(0);
            gradLambdaX = real_t(0);
            gradLambdaY = real_t(0);
            gradLambdaZ = real_t(0);
            return;
        }

        const auto r = std::sqrt(dxSqr);  ///< radial distance from center
        const real_t arg = pi / (real_t(2) * hybridRegionDiameter_) * (r - atomisticRadius_);
        auto base = std::cos(arg);
        lambda = util::powInt(base, exponent_);

        auto factor = -pi / (real_t(2) * hybridRegionDiameter_) * real_c(exponent_) *
                      std::sin(arg) * util::powInt(base, exponent_ - 1) / r;
        gradLambdaX = factor * dx[0];
        gradLambdaY = factor * dx[1];
        gradLambdaZ = factor * dx[2];
    }

    KOKKOS_INLINE_FUNCTION
    real_t operator()(const real_t x, const real_t y, const real_t z) const
    {
        real_t lambda;
        real_t tmp;
        operator()(x, y, z, lambda, tmp, tmp, tmp);
        return lambda;
    }

    Spherical(const std::array<real_t, 3>& center,
              const real_t atomisticRadius,
              const real_t hybridRegionDiameter,
              const int exponent)
        : center_(center),
          atomisticRadius_(atomisticRadius),
          hybridRegionDiameter_(hybridRegionDiameter),
          exponent_(exponent),
          atomisticRadiusSqr_(atomisticRadius * atomisticRadius),
          coarseRadiusSqr_((atomisticRadius + hybridRegionDiameter) *
                           (atomisticRadius + hybridRegionDiameter))
    {
    }
};
}  // namespace weighting_function
}  // namespace mrmd