#pragma once

#include <Kokkos_Core.hpp>
#include <cassert>

#include "data/Particles.hpp"

namespace communication
{
namespace impl
{
class AccumulateForce
{
public:
    void ghostToReal(Particles& particles)
    {
        Particles::force_t::atomic_access_slice force = particles.getForce();
        auto ghost = particles.getGhost();

        auto policy = Kokkos::RangePolicy<>(
            particles.numLocalParticles, particles.numLocalParticles + particles.numGhostParticles);

        auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
        {
            if (ghost(idx) == -1) return;

            auto realIdx = ghost(idx);
            assert(ghost(realIdx) == -1 && "We do not want to add forces to ghost particles!");
            for (auto dim = 0; dim < Particles::DIMENSIONS; ++dim)
            {
                force(realIdx, dim) += force(idx, dim);
                force(idx, dim) = 0_r;
            }
        };

        Kokkos::parallel_for(policy, kernel, "AccumulateForce::ghostToReal");
    }
};

}  // namespace impl
}  // namespace communication