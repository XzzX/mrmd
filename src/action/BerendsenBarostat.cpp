#include "BerendsenBarostat.hpp"

namespace mrmd
{
namespace action
{
namespace BerendsenBarostat
{
void apply(data::Atoms& atoms,
           const real_t& currentPressure,
           const real_t& targetPressure,
           const real_t& gamma,
           data::Subdomain& subdomain)
{
    auto mu = std::cbrt(1_r + gamma * (currentPressure - targetPressure));
    subdomain.scale(mu);

    auto pos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        pos(idx, 0) *= mu;
        pos(idx, 1) *= mu;
        pos(idx, 2) *= mu;
    };
    Kokkos::parallel_for(policy, kernel, "BerendsenBarostat::apply");

    Kokkos::fence();
}

}  // namespace BerendsenBarostat
}  // namespace action
}  // namespace mrmd