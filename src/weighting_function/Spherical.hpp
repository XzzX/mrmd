#pragma once

#include "constants.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"

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

public:
    real_t operator()(const real_t x, const real_t y, const real_t z)
    {
        real_t dx[3] = {x - center_[0], y - center_[1], z - center_[2]};
        auto dxSqr = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];

        if (dxSqr < atomisticRadiusSqr_) return 1_r;
        if (dxSqr > coarseRadiusSqr_) return 0_r;

        auto cos =
            std::cos(pi / (2_r * hybridRegionDiameter_) * (std::sqrt(dxSqr) - atomisticRadius_));
        return cos * cos;
    }
    Spherical(const std::array<real_t, 3>& center,
              const real_t atomisticRadius,
              const real_t hybridRegionDiameter)
        : center_(center),
          atomisticRadius_(atomisticRadius),
          hybridRegionDiameter_(hybridRegionDiameter),
          atomisticRadiusSqr_(atomisticRadius * atomisticRadius),
          coarseRadiusSqr_((atomisticRadius + hybridRegionDiameter) *
                           (atomisticRadius + hybridRegionDiameter))
    {
    }
};
}  // namespace weighting_function
}  // namespace mrmd