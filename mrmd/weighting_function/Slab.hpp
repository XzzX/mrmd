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
    const idx_t interfaceType_;

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
    void setWeightInHY(real_t& dx,
                       real_t& absDx,
                       real_t& lambda,
                       real_t& modulatedLambda,
                       real_t& gradLambdaX,
                       real_t& gradLambdaY,
                       real_t& gradLambdaZ) const
    {
        switch (interfaceType_)
        {
            case 0:
                // smooth interface
                setSmoothWeightInHY(
                    dx, absDx, lambda, modulatedLambda, gradLambdaX, gradLambdaY, gradLambdaZ);
                break;
            case 1:
                // abrupt interface
                setAbruptWeightInHY(lambda, modulatedLambda, gradLambdaX, gradLambdaY, gradLambdaZ);
                break;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void setSmoothWeightInHY(real_t& dx,
                             real_t& absDx,
                             real_t& lambda,
                             real_t& modulatedLambda,
                             real_t& gradLambdaX,
                             real_t& gradLambdaY,
                             real_t& gradLambdaZ) const
    {
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

    KOKKOS_INLINE_FUNCTION
    void setAbruptWeightInHY(real_t& lambda,
                             real_t& modulatedLambda,
                             real_t& gradLambdaX,
                             real_t& gradLambdaY,
                             real_t& gradLambdaZ) const
    {
        // abrupt interface sets weight to the same value as in AT region
        setWeightInAT(lambda, modulatedLambda, gradLambdaX, gradLambdaY, gradLambdaZ);
    }

    KOKKOS_INLINE_FUNCTION
    void setWeightInAT(real_t& lambda,
                       real_t& modulatedLambda,
                       real_t& gradLambdaX,
                       real_t& gradLambdaY,
                       real_t& gradLambdaZ) const
    {
        lambda = 1_r;
        modulatedLambda = 1_r;
        gradLambdaX = 0_r;
        gradLambdaY = 0_r;
        gradLambdaZ = 0_r;
    }

    KOKKOS_INLINE_FUNCTION
    void setWeightInCG(real_t& lambda,
                       real_t& modulatedLambda,
                       real_t& gradLambdaX,
                       real_t& gradLambdaY,
                       real_t& gradLambdaZ) const
    {
        lambda = 0_r;
        modulatedLambda = 0_r;
        gradLambdaX = 0_r;
        gradLambdaY = 0_r;
        gradLambdaZ = 0_r;
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
        real_t dx = x - center_[0];
        real_t absDx = std::abs(dx);

        if (absDx < atomisticRegionHalfDiameter_)
        {
            setWeightInAT(lambda, modulatedLambda, gradLambdaX, gradLambdaY, gradLambdaZ);
            return;
        }
        else if (absDx > atomisticRegionHalfDiameter_ + hybridRegionDiameter_)
        {
            setWeightInCG(lambda, modulatedLambda, gradLambdaX, gradLambdaY, gradLambdaZ);
            return;
        }
        else
        {
            setWeightInHY(
                dx, absDx, lambda, modulatedLambda, gradLambdaX, gradLambdaY, gradLambdaZ);
            return;
        }
    }

    Slab(const Point3D& center,
         const real_t atomisticRegionDiameter,
         const real_t hybridRegionDiameter,
         const idx_t nu,
         const idx_t interfaceType = 0)
        : center_(center),
          atomisticRegionHalfDiameter_(0.5_r * atomisticRegionDiameter),
          hybridRegionDiameter_(hybridRegionDiameter),
          exponent_(2 * nu),
          interfaceType_(interfaceType)
    {
    }
};
}  // namespace weighting_function
}  // namespace mrmd