#include "LimitVelocity.hpp"

#include <Kokkos_Core.hpp>

namespace mrmd
{
namespace action
{
void limitVelocityPerComponent(data::Atoms& atoms, const real_t& maxVelocityPerComponent)
{
    auto vel = atoms.getVel();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        vel(idx, 0) = std::min(vel(idx, 0), +maxVelocityPerComponent);
        vel(idx, 0) = std::max(vel(idx, 0), -maxVelocityPerComponent);

        vel(idx, 1) = std::min(vel(idx, 1), +maxVelocityPerComponent);
        vel(idx, 1) = std::max(vel(idx, 1), -maxVelocityPerComponent);

        vel(idx, 2) = std::min(vel(idx, 2), +maxVelocityPerComponent);
        vel(idx, 2) = std::max(vel(idx, 2), -maxVelocityPerComponent);
    };
    Kokkos::parallel_for("limitVelocityPerComponent", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd