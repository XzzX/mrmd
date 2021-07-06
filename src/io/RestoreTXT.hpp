#pragma once

#include "data/Particles.hpp"

namespace mrmd
{
namespace io
{
data::Particles restoreParticles(const std::string& filename);
}  // namespace io
}  // namespace mrmd