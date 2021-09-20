#pragma once

#include "datatypes.hpp"

namespace mrmd
{
/// number of spatial dimensions
constexpr static int DIMENSIONS = 3;

constexpr real_t pi = 3.14159265358979323846;
constexpr real_t M_SQRTPI = 1.77245385090551602729;  // sqrt(pi)

namespace constants
{
constexpr real_t N_A = 6.02214129e23;  ///< Avogadro constant (1/mol)
constexpr real_t k_b = 8.3144621e-3;   ///< Boltzmann's constant (kJ/mol/K)
}  // namespace constants
}  // namespace mrmd