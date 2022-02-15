#pragma once

#include "data/Atoms.hpp"

namespace mrmd
{
namespace io
{
void dumpCSV(const std::string& filename, data::Atoms& atoms, bool dumpGhosts = true);
}  // namespace io
}  // namespace mrmd