#pragma once

#include "data/Particles.hpp"

namespace mrmd
{
namespace analysis
{
std::array<real_t, 3> getSystemMomentum(data::Particles& particles);

}  // namespace analysis
}  // namespace mrmd