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

#include "constants.hpp"
#include "datatypes.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace weighting_function
{
constexpr real_t REGION_CHECK_EPSILON = 0_r;

KOKKOS_INLINE_FUNCTION bool isInATRegion(const real_t& lambda)
{
    return lambda >= (1_r - REGION_CHECK_EPSILON);
}
KOKKOS_INLINE_FUNCTION bool isInCGRegion(const real_t& lambda)
{
    return lambda <= REGION_CHECK_EPSILON;
}
KOKKOS_INLINE_FUNCTION bool isInHYRegion(const real_t& lambda)
{
    return !isInATRegion(lambda) && !isInCGRegion(lambda);
}

}  // namespace weighting_function
}  // namespace mrmd