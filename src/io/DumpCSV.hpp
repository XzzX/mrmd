#pragma once

#include "data/Particles.hpp"

namespace mrmd
{
namespace io
{
void dumpCSV(const std::string& filename, data::Particles& particles);
}  // namespace io
}  // namespace mrmd