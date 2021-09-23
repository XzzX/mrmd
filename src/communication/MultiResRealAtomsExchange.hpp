#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
/**
 * Intended for single process use. Maps atoms that went
 * out of the domain back in a periodic fashion. Mapping is
 * done based on molecule position and atoms are moved according
 * to their molecule.
 *
 * @pre Atom position is at most one periodic copy away
 * from the subdomain.
 * @post Atom position lies within half-open interval
 * [min, max) for all coordinate dimensions.
 */
void realAtomsExchange(const data::Subdomain& subdomain,
                       const data::Molecules& molecules,
                       const data::Atoms& atoms)
{
    auto moleculesPos = molecules.getPos();
    auto atomsOffset = molecules.getAtomsOffset();
    auto numAtoms = molecules.getNumAtoms();

    auto atomsPos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
    auto kernel = KOKKOS_LAMBDA(const idx_t& moleculeIdx)
    {
        for (auto dim = 0; dim < DIMENSIONS; ++dim)
        {
            auto& moleculeX = moleculesPos(moleculeIdx, dim);
            if (subdomain.maxCorner[dim] <= moleculeX)
            {
                moleculeX -= subdomain.diameter[dim];

                auto atomsStart = atomsOffset(moleculeIdx);
                auto atomsEnd = atomsStart + numAtoms(moleculeIdx);
                for (idx_t atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
                {
                    auto& atomX = atomsPos(atomIdx, dim);
                    atomX -= subdomain.diameter[dim];
                }
            }
            if (moleculeX < subdomain.minCorner[dim])
            {
                moleculeX += subdomain.diameter[dim];

                auto atomsStart = atomsOffset(moleculeIdx);
                auto atomsEnd = atomsStart + numAtoms(moleculeIdx);
                for (idx_t atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
                {
                    auto& atomX = atomsPos(atomIdx, dim);
                    atomX += subdomain.diameter[dim];
                }
            }
            assert(moleculeX < subdomain.maxCorner[dim]);
            assert(subdomain.minCorner[dim] <= moleculeX);
        }
    };
    Kokkos::parallel_for(policy, kernel, "realAtomsExchange::periodicMapping");
    Kokkos::fence();
}

}  // namespace communication
}  // namespace mrmd