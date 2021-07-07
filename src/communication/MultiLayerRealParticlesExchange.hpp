#pragma once

#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
/**
 * Intended for single process use. Maps particles that went
 * out of the domain back in a periodic fashion. Mapping is
 * done based on molecule position and atoms are moved according
 * to their molecule.
 *
 * @pre Particle position is at most one periodic copy away
 * from the subdomain.
 * @post Particle position lies within half-open interval
 * [min, max) for all coordinate dimensions.
 */
void realParticlesExchange(const data::Subdomain& subdomain,
                           const data::Particles& molecules,
                           const data::Particles& atoms)
{
    auto moleculePos = molecules.getPos();
    auto atomPos = atoms.getPos();
    auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        for (auto dim = 0; dim < data::Particles::DIMENSIONS; ++dim)
        {
            auto& moleculeX = moleculePos(idx, dim);
            if (subdomain.maxCorner[dim] <= moleculeX)
            {
                moleculeX -= subdomain.diameter[dim];

                auto& atomX = atomPos(idx, dim);
                atomX -= subdomain.diameter[dim];
            }
            if (moleculeX < subdomain.minCorner[dim])
            {
                moleculeX += subdomain.diameter[dim];

                auto& atomX = atomPos(idx, dim);
                atomX += subdomain.diameter[dim];
            }
            assert(moleculeX < subdomain.maxCorner[dim]);
            assert(subdomain.minCorner[dim] <= moleculeX);
        }
    };
    Kokkos::parallel_for(policy, kernel, "realParticlesExchange::periodicMapping");
    Kokkos::fence();
}

}  // namespace communication
}  // namespace mrmd