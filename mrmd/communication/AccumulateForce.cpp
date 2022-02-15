#include "AccumulateForce.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
namespace AccumulateForce
{
void ghostToReal(data::Atoms& atoms, const IndexView& correspondingRealAtom)
{
    data::Atoms::force_t::atomic_access_slice force = atoms.getForce();

    auto policy =
        Kokkos::RangePolicy<>(atoms.numLocalAtoms, atoms.numLocalAtoms + atoms.numGhostAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        if (correspondingRealAtom(idx) == -1) return;

        auto realIdx = correspondingRealAtom(idx);
        assert(correspondingRealAtom(realIdx) == -1 &&
               "We do not want to add forces to ghost atoms!");
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            force(realIdx, dim) += force(idx, dim);
            force(idx, dim) = 0_r;
        }
    };

    Kokkos::parallel_for(policy, kernel, "AccumulateForce::ghostToReal");
    Kokkos::fence();
}
}  // namespace AccumulateForce
}  // namespace impl
}  // namespace communication
}  // namespace mrmd