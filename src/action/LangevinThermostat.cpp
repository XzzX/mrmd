#include "LangevinThermostat.hpp"

#include <Kokkos_Core.hpp>

namespace mrmd
{
namespace action
{
void LangevinThermostat::apply(data::Particles& particles)
{
    auto vel = particles.getVel();
    auto force = particles.getForce();

    auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        constexpr real_t mass = 1_r;
        constexpr real_t massSqrt = 1_r;  // std::sqrt(mass);

        force(idx, 0) += pref1 * vel(idx, 0) * mass + pref2 * (rng.draw() - 0.5_r) * massSqrt;
        force(idx, 1) += pref1 * vel(idx, 1) * mass + pref2 * (rng.draw() - 0.5_r) * massSqrt;
        force(idx, 2) += pref1 * vel(idx, 2) * mass + pref2 * (rng.draw() - 0.5_r) * massSqrt;
    };
    Kokkos::parallel_for(policy, kernel, "LangevinThermostat::applyThermostat");

    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd