#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"

namespace mrmd
{
namespace io
{
void restoreLAMMPS(const std::string& filename, data::Atoms& atoms);
}  // namespace io
}  // namespace mrmd