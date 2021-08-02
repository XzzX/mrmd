#pragma once

#include "constants.hpp"
#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace weighting_function
{
class Slab
{
private:
    const std::array<real_t, 3> center_;
    const real_t atomisticRegionHalfDiameter_;
    const real_t hybridRegionDiameter_;
    const idx_t exponent_;

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
        auto dx = x - center_[0];
        auto absDx = std::abs(dx);

        if (absDx < atomisticRegionHalfDiameter_)
        {
            lambda = 1_r;
            gradLambdaX = 0_r;
            gradLambdaY = 0_r;
            gradLambdaZ = 0_r;
            return;
        }
        if (absDx > atomisticRegionHalfDiameter_ + hybridRegionDiameter_)
        {
            lambda = 0_r;
            gradLambdaX = 0_r;
            gradLambdaY = 0_r;
            gradLambdaZ = 0_r;
            return;
        }

        const real_t arg =
            pi / (2_r * hybridRegionDiameter_) * (absDx - atomisticRegionHalfDiameter_);
        auto base = std::cos(arg);
        lambda = util::powInt(base, exponent_);

        auto factor = -pi / (2_r * hybridRegionDiameter_) * real_c(exponent_) * std::sin(arg) *
                      util::powInt(base, exponent_ - 1) / absDx;
        gradLambdaX = factor * dx;
        gradLambdaY = 0_r;
        gradLambdaZ = 0_r;
    }

    KOKKOS_INLINE_FUNCTION
    real_t operator()(const real_t x, const real_t y, const real_t z) const
    {
        real_t lambda;
        real_t tmp;
        operator()(x, y, z, lambda, tmp, tmp, tmp);
        return lambda;
    }

    Slab(const std::array<real_t, 3>& center,
         const real_t atomisticRegionDiameter,
         const real_t hybridRegionDiameter,
         const idx_t nu)
        : center_(center),
          atomisticRegionHalfDiameter_(0.5_r * atomisticRegionDiameter),
          hybridRegionDiameter_(hybridRegionDiameter),
          exponent_(2 * nu)
    {
    }
};
}  // namespace weighting_function
}  // namespace mrmd