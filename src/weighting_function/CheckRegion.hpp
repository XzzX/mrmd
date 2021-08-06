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