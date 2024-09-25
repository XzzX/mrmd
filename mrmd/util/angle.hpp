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

namespace mrmd
{
namespace util
{
/**
 * converts degree to radians
 */
KOKKOS_INLINE_FUNCTION
constexpr real_t degToRad(const real_t& grd) { return grd / 180_r * M_PI; }

/**
 * converts radians to degree
 */
KOKKOS_INLINE_FUNCTION
constexpr real_t radToDeg(const real_t& rad) { return rad / M_PI * 180_r; }

}  // namespace util
}  // namespace mrmd
