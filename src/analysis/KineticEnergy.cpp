#include "KineticEnergy.hpp"

namespace mrmd
{
namespace analysis
{
real_t getKineticEnergy(data::Particles& particles)
{
    auto vel = particles.getVel();
    auto mass = particles.getMass();
    real_t velSqr = 0_r;
    auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, real_t& sum)
    {
        sum += mass(idx) *
               (vel(idx, 0) * vel(idx, 0) + vel(idx, 1) * vel(idx, 1) + vel(idx, 2) * vel(idx, 2));
    };
    Kokkos::parallel_reduce("getKineticEnergy", policy, kernel, velSqr);
    return 0.5_r * velSqr;
}

}  // namespace analysis
}  // namespace mrmd