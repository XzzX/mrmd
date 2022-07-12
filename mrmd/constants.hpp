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
constexpr real_t temperature = real_t(1e3) / (real_t(6.02214129) * real_t(1.380649));
constexpr real_t energy = real_t(1e3) / real_t(6.02214129e23);
}  // namespace toSI
}  // namespace mrmd