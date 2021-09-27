#pragma once

#include "datatypes.hpp"

namespace mrmd
{
/// number of spatial dimensions
constexpr static int DIMENSIONS = 3;

constexpr real_t pi = 3.14159265358979323846;
constexpr real_t M_SQRTPI = 1.77245385090551602729;  // sqrt(pi)

namespace toSI
{
constexpr real_t temperature = 1e3_r / (6.02214129_r * 1.380649_r);
constexpr real_t energy = 1e3_r / 6.02214129e23_r;
}  // namespace toSI
}  // namespace mrmd