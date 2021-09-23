#pragma once

#include "data/Molecules.hpp"
#include "data/Atoms.hpp"

namespace mrmd
{
namespace io
{
void restoreLAMMPS(const std::string& filename, data::Atoms& atoms, data::Molecules& molecules);
}  // namespace io
}  // namespace mrmd