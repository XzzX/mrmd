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

/**
 * Find the index of the first bin in the grid that is greater than or equal to the value.
 * @param grid grid to search in
 * @param value value to find
 * @return index of the right bin
 */
KOKKOS_INLINE_FUNCTION
idx_t findRightBin(const ScalarView& grid, const real_t& value)
{
    idx_t idx = 0;
    for (; idx < idx_c(grid.extent(0)) && grid(idx) < value; ++idx)
    {
        // empty loop body
    }

    return idx;
}

/**
 * Interpolate data values of MultiHistogram to a new grid.
 * @param input input MultiHistogram
 * @param grid grid to interpolate to
 * @return MultiHistogram with interpolated values
 */
data::MultiHistogram interpolate(const data::MultiHistogram& input, const ScalarView& grid);

data::MultiHistogram constrainToSymmetricSlab(const data::MultiHistogram& input,
                                                  const util::IsInSymmetricSlab& applicationRegion);

}  // namespace util
}  // namespace mrmd