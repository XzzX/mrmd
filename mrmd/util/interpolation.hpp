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

#include <limits>

#include "data/MultiHistogram.hpp"
#include "util/IsInSymmetricSlab.hpp"

namespace mrmd
{
namespace util
{
/**
 * Linear interpolation between two values.
 * @param left left value
 * @param right right value
 * @param factor interpolation factor in [0, 1]
 * @return interpolated value
 */
KOKKOS_INLINE_FUNCTION
real_t lerp(const real_t& left, const real_t& right, const real_t& factor)
{
    return left + (right - left) * factor;
}

KOKKOS_INLINE_FUNCTION
bool isFloatEQ(const real_t& a,
               const real_t& b,
               const real_t& epsilon = std::numeric_limits<real_t>::epsilon())
{
    return std::abs(a - b) <= epsilon;
}

/**
 * Linear interpolation of data contained in input MultiHistogram onto grid of target histogram,
 * adding the interpolated data to the data already contained in target. Data for target grid points
 * outside of the grid range of the input MultiHistogram is not updated.
 * @param input MultiHistogram containing data to interpolate on coarse grid.
 * @param target MultiHistogram defining the grid to interpolate onto and containing the data
 * to be updated by interpolating the data from input.
 */
void updateInterpolate(const data::MultiHistogram& target, const data::MultiHistogram& input);

}  // namespace util
}  // namespace mrmd