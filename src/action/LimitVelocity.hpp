#pragma once

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
void limitVelocityPerComponent(data::Atoms& atoms, const real_t& maxVelocityPerComponent);
}  // namespace action
}  // namespace mrmd