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
    void ghostToReal(data::Particles& particles, IndexView correspondingRealParticle)
    {
        data::Particles::force_t::atomic_access_slice force = particles.getForce();

        auto policy = Kokkos::RangePolicy<>(
            particles.numLocalParticles, particles.numLocalParticles + particles.numGhostParticles);

        auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
        {
            if (correspondingRealParticle(idx) == -1) return;

            auto realIdx = correspondingRealParticle(idx);
            assert(correspondingRealParticle(realIdx) == -1 &&
                   "We do not want to add forces to ghost particles!");
            for (auto dim = 0; dim < data::Particles::DIMENSIONS; ++dim)
            {
                force(realIdx, dim) += force(idx, dim);
                force(idx, dim) = 0_r;
            }
        };

        Kokkos::parallel_for(policy, kernel, "AccumulateForce::ghostToReal");
        Kokkos::fence();
    }
};

}  // namespace impl
}  // namespace communication