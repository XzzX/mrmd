#pragma once

#include "data/Molecules.hpp"
#include "data/Particles.hpp"

namespace mrmd
{
namespace io
{
void restoreLAMMPS(const std::string& filename, data::Particles& atoms, data::Molecules& molecules);
}  // namespace io
}  // namespace mrmd