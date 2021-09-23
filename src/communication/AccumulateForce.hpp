#pragma once

#include <Kokkos_Core.hpp>
#include <cassert>

#include "data/Atoms.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class AccumulateForce
{
private:
    data::Atoms::force_t::atomic_access_slice force_;
    IndexView correspondingRealAtom_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        if (correspondingRealAtom_(idx) == -1) return;

        auto realIdx = correspondingRealAtom_(idx);
        assert(correspondingRealAtom_(realIdx) == -1 &&
               "We do not want to add forces to ghost atoms!");
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            force_(realIdx, dim) += force_(idx, dim);
            force_(idx, dim) = 0_r;
        }
    }

    void ghostToReal(data::Atoms& atoms, const IndexView& correspondingRealAtom)
    {
        force_ = atoms.getForce();
        correspondingRealAtom_ = correspondingRealAtom;

        auto policy =
            Kokkos::RangePolicy<>(atoms.numLocalAtoms, atoms.numLocalAtoms + atoms.numGhostAtoms);

        Kokkos::parallel_for(policy, *this, "AccumulateForce::ghostToReal");
        Kokkos::fence();
    }
};

}  // namespace impl
}  // namespace communication
}  // namespace mrmd