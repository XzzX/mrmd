#pragma once

#include <Kokkos_Core.hpp>

#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class VelocityVerlet
{
public:
    static real_t preForceIntegrate(data::Particles& particles, const real_t dt);
    static void postForceIntegrate(data::Particles& particles, const real_t dt);
};
}  // namespace action
}  // namespace mrmd