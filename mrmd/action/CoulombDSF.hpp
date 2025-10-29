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
    real_t rc_ = 0_r;
    real_t rcSqr_ = 0_r;
    real_t forceShift_ = 0_r;
    real_t energyShift_ = 0_r;

public:
    /**
     *  DOI: 10.1140/epjst/e2016-60151-6
     *  eq 16
     */
    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        auto r = std::sqrt(distSqr);
        real_t prefac = 138.935458_r * q1 * q2;
        real_t expX2;
        auto erfc = util::approxErfc(alpha_ * r, expX2);
        auto force = prefac * (erfc / r + 2_r * alpha_ * inv_sqrtpi * expX2 + r * forceShift_);
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
        real_t prefac = 138.935458_r * q1 * q2;
        auto erfc = util::approxErfc(alpha_ * r);
        return prefac * (erfc / r - energyShift_ - forceShift_ * (r - rc_));
    }

    CoulombDSF(const real_t& rc, const real_t& alpha) : alpha_(alpha), rc_(rc), rcSqr_(rc * rc)
    {
        real_t erfc = std::erfc(alpha_ * rc);
        real_t exp = std::exp(-alpha_ * alpha_ * rcSqr_);
        forceShift_ = -(erfc / rcSqr_ + 2_r * inv_sqrtpi * alpha_ * exp / rc);
        energyShift_ = erfc / rc;
    }
};
}  // namespace impl

}  // namespace action
}  // namespace mrmd