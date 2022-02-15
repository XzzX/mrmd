#pragma once

#include "data/Atoms.hpp"

namespace mrmd
{
namespace analysis
{
std::array<real_t, 3> getSystemMomentum(data::Atoms& atoms);

}  // namespace analysis
}  // namespace mrmd