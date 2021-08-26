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
