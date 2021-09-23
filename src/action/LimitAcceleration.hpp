#pragma once

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
void limitAccelerationPerComponent(data::Atoms& atoms, const real_t& maxAccelerationPerComponent);
}  // namespace action
}  // namespace mrmd