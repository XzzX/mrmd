#pragma once

#include "data/Atoms.hpp"

namespace mrmd
{
namespace analysis
{
/**
 * @return total kinetic energy of all local atoms
 */
real_t getKineticEnergy(data::Atoms& atoms);

/**
 * @return average kinetic energy per local atom
 */
real_t getMeanKineticEnergy(data::Atoms& atoms);
}  // namespace analysis
}  // namespace mrmd