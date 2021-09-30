#pragma once

#include <Kokkos_Random.hpp>

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class VelocityVerletLangevinThermostat
{
private:
    Kokkos::Random_XorShift1024_Pool<> randPool_ = Kokkos::Random_XorShift1024_Pool<>(1234);
    real_t zeta_;
    real_t temperature_;

public:
    void set(const real_t& zeta, const real_t& temperature)
    {
        zeta_ = zeta;
        temperature_ = temperature;
    }

    VelocityVerletLangevinThermostat(const real_t& zeta, const real_t& temperature)
    {
        set(zeta, temperature);
    }

    real_t preForceIntegrate(data::Atoms& atoms, const real_t dt);
    void postForceIntegrate(data::Atoms& atoms, const real_t dt);
};
}  // namespace action
}  // namespace mrmd