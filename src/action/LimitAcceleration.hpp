#pragma once

#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
void limitAccelerationPerComponent(data::Particles& atoms,
                                   const real_t& maxAccelerationPerComponent);
}  // namespace action
}  // namespace mrmd