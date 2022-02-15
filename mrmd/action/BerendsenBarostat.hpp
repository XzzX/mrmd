#pragma once

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
namespace BerendsenBarostat
{
/**
 * Berendsen Barostat
 * DOI: 10.1063/1.448118
 */
template <bool stretchX = true, bool stretchY = true, bool stretchZ = true>
void apply(data::Atoms& atoms,
           const real_t& currentPressure,
           const real_t& targetPressure,
           const real_t& gamma,
           data::Subdomain& subdomain)
{
    auto mu = std::cbrt(1_r + gamma * (currentPressure - targetPressure));
    if constexpr (stretchX) subdomain.scaleDim(mu, COORD_X);
    if constexpr (stretchY) subdomain.scaleDim(mu, COORD_Y);
    if constexpr (stretchZ) subdomain.scaleDim(mu, COORD_Z);

    auto pos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        if constexpr (stretchX) pos(idx, 0) *= mu;
        if constexpr (stretchY) pos(idx, 1) *= mu;
        if constexpr (stretchZ) pos(idx, 2) *= mu;
    };
    Kokkos::parallel_for(policy, kernel, "BerendsenBarostat::apply");

    Kokkos::fence();
}
}  // namespace BerendsenBarostat
}  // namespace action
}  // namespace mrmd