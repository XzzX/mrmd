#include <Kokkos_Core.hpp>

#include "VelocityVerlet.hpp"

namespace mrmd
{
namespace action
{
void limitAccelerationPerComponent(data::Particles& atoms,
                                   const real_t& maxAccelerationPerComponent)
{
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        auto m = mass(idx);
        auto invM = 1_r / m;

        force(idx, 0) = std::min(force(idx, 0) * invM, +maxAccelerationPerComponent) * m;
        force(idx, 0) = std::max(force(idx, 0) * invM, -maxAccelerationPerComponent) * m;

        force(idx, 1) = std::min(force(idx, 1) * invM, +maxAccelerationPerComponent) * m;
        force(idx, 1) = std::max(force(idx, 1) * invM, -maxAccelerationPerComponent) * m;

        force(idx, 2) = std::min(force(idx, 2) * invM, +maxAccelerationPerComponent) * m;
        force(idx, 2) = std::max(force(idx, 2) * invM, -maxAccelerationPerComponent) * m;
    };
    Kokkos::parallel_for("limitForcePerComponent", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd