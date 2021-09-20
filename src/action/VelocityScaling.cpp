#include "VelocityScaling.hpp"

#include <Kokkos_Core.hpp>

#include "analysis/KineticEnergy.hpp"

namespace mrmd
{
namespace action
{
void VelocityScaling::apply(data::Particles& particles) const
{
    auto Ekin = analysis::getKineticEnergy(particles);
    auto T = Ekin * 2_r / (3_r * real_c(particles.numLocalParticles));
    auto beta = std::sqrt(1_r + gamma_ * (targetTemperature_ / T - 1_r));

    auto vel = particles.getVel();

    auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        vel(idx, 0) *= beta;
        vel(idx, 1) *= beta;
        vel(idx, 2) *= beta;
    };
    Kokkos::parallel_for(policy, kernel, "VelocityScaling::apply");

    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd