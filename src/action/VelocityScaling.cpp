#include "VelocityScaling.hpp"

#include <Kokkos_Core.hpp>

#include "analysis/KineticEnergy.hpp"

namespace mrmd
{
namespace action
{
void VelocityScaling::apply(data::Atoms& atoms, const real_t& degreesOfFreedomPerAtom) const
{
    auto Ekin = analysis::getKineticEnergy(atoms);
    auto T = Ekin * 2_r / (degreesOfFreedomPerAtom * real_c(atoms.numLocalAtoms));
    auto beta = std::sqrt(1_r + gamma_ * (targetTemperature_ / T - 1_r));

    auto vel = atoms.getVel();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
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