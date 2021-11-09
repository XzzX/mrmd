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
class Coulomb
{
public:
    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        real_t prefac = 138.935458_r * q1 * q2;
        return prefac / distSqr;
    }

    KOKKOS_INLINE_FUNCTION
    real_t computeEnergy(const real_t& distSqr, const real_t q1, const real_t q2) const
    {
        auto r = std::sqrt(distSqr);
        real_t prefac = 138.935458_r * q1 * q2;
        return prefac / r;
    }
};
}  // namespace impl

}  // namespace action
}  // namespace mrmd