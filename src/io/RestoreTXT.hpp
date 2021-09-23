#pragma once

#include "data/Atoms.hpp"

namespace mrmd
{
namespace io
{
data::Atoms restoreAtoms(const std::string& filename);
}  // namespace io
}  // namespace mrmd