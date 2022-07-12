#pragma once

#include "assert/assert.hpp"
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
    bool isInATRegion(const real_t& x, const real_t& /*y*/, const real_t& /*z*/) const
    {
        auto dx = x - center_[0];
        auto absDx = std::abs(dx);

        return (absDx < atomisticRegionHalfDiameter_);
    }
    KOKKOS_INLINE_FUNCTION
    bool isInHYRegion(const real_t& x, const real_t& /*y*/, const real_t& /*z*/) const
    {
        auto dx = x - center_[0];
        auto absDx = std::abs(dx);

        if (absDx < atomisticRegionHalfDiameter_)
        {
            return false;
        }
        return (absDx < atomisticRegionHalfDiameter_ + hybridRegionDiameter_);
    }
    KOKKOS_INLINE_FUNCTION
    bool isInCGRegion(const real_t& x, const real_t& /*y*/, const real_t& /*z*/) const
    {
        auto dx = x - center_[0];
        auto absDx = std::abs(dx);

        return !(absDx < atomisticRegionHalfDiameter_ + hybridRegionDiameter_);
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const real_t x,
                    const real_t /*y*/,
                    const real_t /*z*/,
                    real_t& lambda,
                    real_t& modulatedLambda,
                    real_t& gradLambdaX,
                    real_t& gradLambdaY,
                    real_t& gradLambdaZ) const
    {
        auto dx = x - center_[0];
        auto absDx = std::abs(dx);

        if (absDx < atomisticRegionHalfDiameter_)
        {
            lambda = real_t(1);
            modulatedLambda = real_t(1);
            gradLambdaX = real_t(0);
            gradLambdaY = real_t(0);
            gradLambdaZ = real_t(0);
            return;
        }
        if (absDx > atomisticRegionHalfDiameter_ + hybridRegionDiameter_)
        {
            lambda = real_t(0);
            modulatedLambda = real_t(0);
            gradLambdaX = real_t(0);
            gradLambdaY = real_t(0);
            gradLambdaZ = real_t(0);
            return;
        }

        const real_t arg =
            pi / (real_t(2) * hybridRegionDiameter_) * (absDx - atomisticRegionHalfDiameter_);
        auto base = std::cos(arg);
        lambda = base * base;
        MRMD_DEVICE_ASSERT(!std::isnan(lambda), "absDx: " << absDx);
        modulatedLambda = util::powInt(base, exponent_);

        auto factor = -pi / (real_t(2) * hybridRegionDiameter_) * real_c(exponent_) *
                      std::sin(arg) * util::powInt(base, exponent_ - 1) / absDx;
        MRMD_DEVICE_ASSERT(!std::isnan(factor));
        MRMD_DEVICE_ASSERT(!std::isnan(dx));
        gradLambdaX = factor * dx;
        gradLambdaY = real_t(0);
        gradLambdaZ = real_t(0);
    }

    Slab(const std::array<real_t, 3>& center,
         const real_t atomisticRegionDiameter,
         const real_t hybridRegionDiameter,
         const idx_t nu)
        : center_(center),
          atomisticRegionHalfDiameter_(real_t(0.5) * atomisticRegionDiameter),
          hybridRegionDiameter_(hybridRegionDiameter),
          exponent_(2 * nu)
    {
    }
};
}  // namespace weighting_function
}  // namespace mrmd