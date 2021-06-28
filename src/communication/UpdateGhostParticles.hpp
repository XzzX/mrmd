#pragma once

#include <Kokkos_Core.hpp>
#include <cassert>

#include "data/Particles.hpp"

namespace communication
{
namespace impl
{
class UpdateGhostParticles
{
private:
    Particles::pos_t pos_;
    Subdomain subdomain_;
    IndexView correspondingRealParticle_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t idx) const
    {
        auto realIdx = correspondingRealParticle_(idx);

        assert(realIdx != idx);
        assert(correspondingRealParticle_(realIdx) == -1);

        if (pos_(idx, 0) >= pos_(realIdx, 0))
            pos_(idx, 0) = pos_(realIdx, 0) + subdomain_.diameter[0];
        if (pos_(idx, 1) >= pos_(realIdx, 1))
            pos_(idx, 1) = pos_(realIdx, 1) + subdomain_.diameter[1];
        if (pos_(idx, 2) >= pos_(realIdx, 2))
            pos_(idx, 2) = pos_(realIdx, 2) + subdomain_.diameter[2];
        if (pos_(idx, 0) < pos_(realIdx, 0))
            pos_(idx, 0) = pos_(realIdx, 0) - subdomain_.diameter[0];
        if (pos_(idx, 1) < pos_(realIdx, 1))
            pos_(idx, 1) = pos_(realIdx, 1) - subdomain_.diameter[1];
        if (pos_(idx, 2) < pos_(realIdx, 2))
            pos_(idx, 2) = pos_(realIdx, 2) - subdomain_.diameter[2];
    }

    void updateOnlyPos(Particles& particles, IndexView correspondingRealParticle)
    {
        pos_ = particles.getPos();
        correspondingRealParticle_ = correspondingRealParticle;

        auto policy = Kokkos::RangePolicy<>(
            particles.numLocalParticles, particles.numLocalParticles + particles.numGhostParticles);
        Kokkos::parallel_for(policy, *this, "UpdateGhostParticles::updateOnlyPos");
    }

    UpdateGhostParticles(const Subdomain& subdomain) : subdomain_(subdomain) {}
};

}  // namespace impl
}  // namespace communication