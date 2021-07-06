#pragma once

#include <Kokkos_Core.hpp>
#include <cassert>

#include "data/Particles.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class AccumulateForce
{
private:
    data::Particles::force_t::atomic_access_slice force_;
    IndexView correspondingRealParticle_;

public:
    void operator()(const idx_t& idx) const
    {
        if (correspondingRealParticle_(idx) == -1) return;

        auto realIdx = correspondingRealParticle_(idx);
        assert(correspondingRealParticle_(realIdx) == -1 &&
               "We do not want to add forces to ghost particles!");
        for (auto dim = 0; dim < data::Particles::DIMENSIONS; ++dim)
        {
            force_(realIdx, dim) += force_(idx, dim);
            force_(idx, dim) = 0_r;
        }
    }

    void ghostToReal(data::Particles& particles, IndexView correspondingRealParticle)
    {
        force_ = particles.getForce();
        correspondingRealParticle_ = correspondingRealParticle;

        auto policy = Kokkos::RangePolicy<>(
            particles.numLocalParticles, particles.numLocalParticles + particles.numGhostParticles);

        Kokkos::parallel_for(policy, *this, "AccumulateForce::ghostToReal");
        Kokkos::fence();
    }
};

}  // namespace impl
}  // namespace communication
}  // namespace mrmd