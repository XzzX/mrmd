#include "BerendsenThermostat.hpp"

#include <Kokkos_Core.hpp>

#include "assert.hpp"

namespace mrmd
{
namespace action
{
namespace BerendsenThermostat
{
void apply(data::Atoms& atoms,
           const real_t& currentTemperature,
           const real_t& targetTemperature,
           const real_t& gamma)
{
    ASSERT(currentTemperature > 0_r);
    ASSERT(targetTemperature > 0_r);
    
    auto beta = std::sqrt(1_r + gamma * (targetTemperature / currentTemperature - 1_r));

    auto vel = atoms.getVel();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        vel(idx, 0) *= beta;
        vel(idx, 1) *= beta;
        vel(idx, 2) *= beta;
    };
    Kokkos::parallel_for(policy, kernel, "BerendsenThermostat::apply");

    Kokkos::fence();
}
}  // namespace BerendsenThermostat
}  // namespace action
}  // namespace mrmd