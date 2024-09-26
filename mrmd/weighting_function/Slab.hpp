// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    const Point3D center_;
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
            lambda = 1_r;
            modulatedLambda = 1_r;
            gradLambdaX = 0_r;
            gradLambdaY = 0_r;
            gradLambdaZ = 0_r;
            return;
        }
        if (absDx > atomisticRegionHalfDiameter_ + hybridRegionDiameter_)
        {
            lambda = 0_r;
            modulatedLambda = 0_r;
            gradLambdaX = 0_r;
            gradLambdaY = 0_r;
            gradLambdaZ = 0_r;
            return;
        }

        const real_t arg =
            pi / (2_r * hybridRegionDiameter_) * (absDx - atomisticRegionHalfDiameter_);
        auto base = std::cos(arg);
        lambda = base * base;
        MRMD_DEVICE_ASSERT(!std::isnan(lambda), "absDx: " << absDx);
        modulatedLambda = util::powInt(base, exponent_);

        auto factor = -pi / (2_r * hybridRegionDiameter_) * real_c(exponent_) * std::sin(arg) *
                      util::powInt(base, exponent_ - 1) / absDx;
        MRMD_DEVICE_ASSERT(!std::isnan(factor));
        MRMD_DEVICE_ASSERT(!std::isnan(dx));
        gradLambdaX = factor * dx;
        gradLambdaY = 0_r;
        gradLambdaZ = 0_r;
    }

    Slab(const Point3D& center,
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