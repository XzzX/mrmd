#include "LennardJones.hpp"

namespace mrmd::action
{
void LennardJones::apply(data::Atoms& atoms, HalfVerletList& verletList)
{
    energyAndVirial_ = data::EnergyAndVirialReducer();

    pos_ = atoms.getPos();
    force_ = atoms.getForce();
    type_ = atoms.getType();
    verletList_ = verletList;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    Kokkos::parallel_reduce("LennardJones::applyForces", policy, *this, energyAndVirial_);
    Kokkos::fence();
}

LennardJones::LennardJones(const real_t rc,
                           const real_t& sigma,
                           const real_t& epsilon,
                           const real_t& cappingDistance)
    : LennardJones({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, false)
{
}

LennardJones::LennardJones(const std::vector<real_t>& cappingDistance,
                           const std::vector<real_t>& rc,
                           const std::vector<real_t>& sigma,
                           const std::vector<real_t>& epsilon,
                           const idx_t& numTypes,
                           const bool isShifted)
    : LJ_(cappingDistance, rc, sigma, epsilon, numTypes, isShifted), numTypes_(1)
{
    auto rcMax = *std::max_element(rc.begin(), rc.end());
    rcSqr_ = rcMax * rcMax;
}
}  // namespace mrmd::action