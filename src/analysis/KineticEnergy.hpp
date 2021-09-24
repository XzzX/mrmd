#pragma once

#include "data/Atoms.hpp"

namespace mrmd
{
namespace analysis
{
/**
 * @return average kinetic energy per atom
 */
real_t getKineticEnergy(data::Atoms& atoms);
}  // namespace analysis
}  // namespace mrmd