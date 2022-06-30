#include "NoForce.hpp"

namespace mrmd::action
{
void NoForce::apply(data::Atoms& atoms, VerletList& verletList)
{
    pos_ = atoms.getPos();
    force_ = atoms.getForce();
    type_ = atoms.getType();
    verletList_ = verletList;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    Kokkos::parallel_for("NoForce::applyForces", policy, *this);
    Kokkos::fence();
}
}  // namespace mrmd::action