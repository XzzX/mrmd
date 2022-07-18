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
void apply(data::Atoms& atoms,
           const real_t& currentPressure,
           const real_t& targetPressure,
           const real_t& gamma,
           data::Subdomain& subdomain,
           bool stretchX = true,
           bool stretchY = true,
           bool stretchZ = true)
{
    auto mu = std::cbrt(1_r + gamma * (currentPressure - targetPressure));
    if (stretchX) subdomain.scaleDim(mu, COORD_X);
    if (stretchY) subdomain.scaleDim(mu, COORD_Y);
    if (stretchZ) subdomain.scaleDim(mu, COORD_Z);

    auto pos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        if (stretchX) pos(idx, COORD_X) *= mu;
        if (stretchY) pos(idx, COORD_Y) *= mu;
        if (stretchZ) pos(idx, COORD_Z) *= mu;
    };
    Kokkos::parallel_for("BerendsenBarostat::apply", policy, kernel);

    Kokkos::fence();
}
}  // namespace BerendsenBarostat
}  // namespace action
}  // namespace mrmd