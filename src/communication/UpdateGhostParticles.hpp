#pragma once

#include <Kokkos_Core.hpp>
#include <cassert>

#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace communication
{
namespace impl
{
class UpdateGhostParticles
{
private:
    data::Particles::pos_t pos_;
    data::Subdomain subdomain_;
    IndexView correspondingRealParticle_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t idx) const
    {
        auto realIdx = correspondingRealParticle_(idx);

        assert(realIdx != idx);
        assert(correspondingRealParticle_(realIdx) == -1);

        real_t dx[3];
        dx[0] = pos_(idx, 0) - pos_(realIdx, 0);
        dx[1] = pos_(idx, 1) - pos_(realIdx, 1);
        dx[2] = pos_(idx, 2) - pos_(realIdx, 2);

        pos_(idx, 0) = pos_(realIdx, 0);
        pos_(idx, 1) = pos_(realIdx, 1);
        pos_(idx, 2) = pos_(realIdx, 2);

        real_t delta[3];
        delta[0] = 0.1_r * subdomain_.diameter[0];
        delta[1] = 0.1_r * subdomain_.diameter[1];
        delta[2] = 0.1_r * subdomain_.diameter[2];
        if (dx[0] > +delta[0]) pos_(idx, 0) += subdomain_.diameter[0];
        if (dx[1] > +delta[1]) pos_(idx, 1) += subdomain_.diameter[1];
        if (dx[2] > +delta[2]) pos_(idx, 2) += subdomain_.diameter[2];
        if (dx[0] < -delta[0]) pos_(idx, 0) -= subdomain_.diameter[0];
        if (dx[1] < -delta[1]) pos_(idx, 1) -= subdomain_.diameter[1];
        if (dx[2] < -delta[2]) pos_(idx, 2) -= subdomain_.diameter[2];
    }

    void updateOnlyPos(data::Particles& particles, IndexView correspondingRealParticle)
    {
        pos_ = particles.getPos();
        correspondingRealParticle_ = correspondingRealParticle;

        auto policy = Kokkos::RangePolicy<>(
            particles.numLocalParticles, particles.numLocalParticles + particles.numGhostParticles);
        Kokkos::parallel_for(policy, *this, "UpdateGhostParticles::updateOnlyPos");
        Kokkos::fence();
    }

    UpdateGhostParticles(const data::Subdomain& subdomain) : subdomain_(subdomain) {}
};

}  // namespace impl
}  // namespace communication