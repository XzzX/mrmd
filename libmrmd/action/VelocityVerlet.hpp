#pragma once

#include <Kokkos_Core.hpp>

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class VelocityVerlet
{
public:
    static real_t preForceIntegrate(data::Atoms& atoms, const real_t dt);
    static void postForceIntegrate(data::Atoms& atoms, const real_t dt);
};
}  // namespace action
}  // namespace mrmd