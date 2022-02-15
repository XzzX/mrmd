#include "SystemMomentum.hpp"

namespace mrmd
{
namespace analysis
{
std::array<real_t, 3> getSystemMomentum(data::Atoms& atoms)
{
    auto vel = atoms.getVel();
    std::array<real_t, 3> velSum = {0_r, 0_r, 0_r};
    idx_t dim = 0;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    dim = 0;
    Kokkos::parallel_reduce(
        "getSystemMomentum",
        policy,
        KOKKOS_LAMBDA(const idx_t idx, real_t& sum) { sum += vel(idx, dim); },
        velSum[dim]);
    dim = 1;
    Kokkos::parallel_reduce(
        "getSystemMomentum",
        policy,
        KOKKOS_LAMBDA(const idx_t idx, real_t& sum) { sum += vel(idx, dim); },
        velSum[dim]);
    dim = 2;
    Kokkos::parallel_reduce(
        "getSystemMomentum",
        policy,
        KOKKOS_LAMBDA(const idx_t idx, real_t& sum) { sum += vel(idx, dim); },
        velSum[dim]);

    return velSum;
}

}  // namespace analysis
}  // namespace mrmd