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
    void getLambda(const real_t x,
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
            lambda = 1_r;
            gradLambdaX = 0_r;
            gradLambdaY = 0_r;
            gradLambdaZ = 0_r;
            return;
        }
        if (dxSqr > coarseRadiusSqr_)
        {
            lambda = 0_r;
            gradLambdaX = 0_r;
            gradLambdaY = 0_r;
            gradLambdaZ = 0_r;
            return;
        }

        const auto r = std::sqrt(dxSqr);  ///< radial distance from center
        const real_t arg = pi / (2_r * hybridRegionDiameter_) * (r - atomisticRadius_);
        auto base = std::cos(arg);
        lambda = util::powInt(base, exponent_);

        auto factor = -pi / (2_r * hybridRegionDiameter_) * real_c(exponent_) * std::sin(arg) *
                      util::powInt(base, exponent_ - 1) / r;
        gradLambdaX = factor * dx[0];
        gradLambdaY = factor * dx[1];
        gradLambdaZ = factor * dx[2];
    }

    KOKKOS_INLINE_FUNCTION
    real_t operator()(const real_t x, const real_t y, const real_t z) const
    {
        real_t lambda;
        real_t tmp;
        getLambda(x, y, z, lambda, tmp, tmp, tmp);
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