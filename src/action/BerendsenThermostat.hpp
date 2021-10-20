#pragma once

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
namespace BerendsenThermostat
{
/**
 * Berendsen Thermostat
 * DOI: 10.1063/1.448118
 */
void apply(data::Atoms& atoms,
           const real_t& currentTemperature,
           const real_t& targetTemperature,
           const real_t& gamma);
}  // namespace BerendsenThermostat
}  // namespace action
}  // namespace mrmd