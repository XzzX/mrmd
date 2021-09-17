#pragma once

#include <Kokkos_Random.hpp>

#include "data/Particles.hpp"
#include "datatypes.hpp"
#include "util/Random.hpp"

namespace mrmd
{
namespace action
{
class LangevinThermostat
{
private:
    Kokkos::Random_XorShift1024_Pool<> randPool_ = Kokkos::Random_XorShift1024_Pool<>(1234);
    real_t pref1;
    real_t pref2;

public:
    auto getPref1() const { return pref1; }
    auto getPref2() const { return pref2; }

    void apply(data::Particles& particles);

    void set(const real_t gamma, const real_t temperature, const real_t timestep)
    {
        pref1 = -gamma;
        pref2 = std::sqrt(24_r * temperature * gamma / timestep);
    }

    LangevinThermostat(const real_t gamma, const real_t temperature, const real_t timestep)
    {
        set(gamma, temperature, timestep);
    }
};
}  // namespace action
}  // namespace mrmd