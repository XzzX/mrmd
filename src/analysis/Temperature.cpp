#include "Temperature.hpp"

namespace mrmd
{
namespace analysis
{
real_t getTemperature(data::Particles& particles)
{
    auto vel = particles.getVel();
    real_t velSqr = 0_r;
    auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, real_t& sum)
    {
        sum += vel(idx, 0) * vel(idx, 0) + vel(idx, 1) * vel(idx, 1) + vel(idx, 2) * vel(idx, 2);
    };
    Kokkos::parallel_reduce("getTemperature", policy, kernel, velSqr);
    return velSqr / (3_r * real_c(particles.numLocalParticles));
}

}  // namespace analysis
}  // namespace mrmd