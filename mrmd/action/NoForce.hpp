#pragma once

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd::action
{
class NoForce
{
private:
    data::Atoms::pos_t pos_;
    data::Atoms::force_t::atomic_access_slice force_;
    data::Atoms::type_t type_;

    HalfVerletList verletList_;

public:
    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& /*distSqr*/) const { /* TO BE IMPLEMENTED */ }

    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        real_t posTmp[3];
        posTmp[0] = pos_(idx, 0);
        posTmp[1] = pos_(idx, 1);
        posTmp[2] = pos_(idx, 2);

        real_t forceTmp[3] = {0_r, 0_r, 0_r};

        const auto numNeighbors = idx_c(HalfNeighborList::numNeighbor(verletList_, idx));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            idx_t jdx = idx_c(HalfNeighborList::getNeighbor(verletList_, idx, n));
            assert(0 <= jdx);

            auto dx = posTmp[0] - pos_(jdx, 0);
            auto dy = posTmp[1] - pos_(jdx, 1);
            auto dz = posTmp[2] - pos_(jdx, 2);

            auto distSqr = dx * dx + dy * dy + dz * dz;

            auto force = computeForce(distSqr);

            forceTmp[0] += dx * force;
            forceTmp[1] += dy * force;
            forceTmp[2] += dz * force;

            force_(jdx, 0) -= dx * force;
            force_(jdx, 1) -= dy * force;
            force_(jdx, 2) -= dz * force;
        }

        force_(idx, 0) += forceTmp[0];
        force_(idx, 1) += forceTmp[1];
        force_(idx, 2) += forceTmp[2];
    }

    void apply(data::Atoms& atoms, HalfVerletList& verletList);

    NoForce() = default;
};
}  // namespace mrmd::action