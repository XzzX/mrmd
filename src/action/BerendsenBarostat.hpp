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
           data::Subdomain& subdomain);
}  // namespace BerendsenBarostat
}  // namespace action
}  // namespace mrmd