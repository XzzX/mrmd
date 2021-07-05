#pragma once

#include "data/Particles.hpp"

namespace io
{
data::Particles restoreParticles(const std::string& filename);
}  // namespace io