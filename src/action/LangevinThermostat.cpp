#include "LangevinThermostat.hpp"

#include <Kokkos_Core.hpp>

namespace mrmd
{
namespace action
{
void LangevinThermostat::apply(data::Particles& particles)
{
    auto RNG = randPool_;
    auto vel = particles.getVel();
    auto force = particles.getForce();
    auto mass = particles.getMass();
    auto p1 = pref1;  // avoid capturing this pointer
    auto p2 = pref2;  // avoid capturing this pointer

    auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        const real_t m = mass(idx);
        const real_t mSqrt = std::sqrt(m);

        // Get a random number state from the pool for the active thread
        auto randGen = RNG.get_state();

        force(idx, 0) += p1 * vel(idx, 0) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;
        force(idx, 1) += p1 * vel(idx, 1) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;
        force(idx, 2) += p1 * vel(idx, 2) * m + p2 * (randGen.drand() - 0.5_r) * mSqrt;

        // Give the state back, which will allow another thread to acquire it
        RNG.free_state(randGen);
    };
    Kokkos::parallel_for(policy, kernel, "LangevinThermostat::applyThermostat");

    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd