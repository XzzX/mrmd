#pragma once

#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
void limitVelocityPerComponent(data::Particles& atoms, const real_t& maxVelocityPerComponent);
}  // namespace action
}  // namespace mrmd