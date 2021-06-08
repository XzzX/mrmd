#pragma once

#include "Particles.hpp"
#include "Subdomain.hpp"

class HaloExchange
{
private:
    Particles& particles_;
    const Subdomain& subdomain_;

public:
    struct TagX
    {
    };
    struct TagY
    {
    };
    struct TagZ
    {
    };

    KOKKOS_INLINE_FUNCTION
    void copySelf(const idx_t idx, const idx_t dim) const
    {
        if (particles_.getPos(idx, dim) > subdomain_.maxInnerCorner[dim])
        {
            auto nextParticle = particles_.numLocalParticles + particles_.numGhostParticles;
            particles_.copy(idx, nextParticle);
            particles_.getPos(nextParticle, dim) -= subdomain_.diameter[dim];
            auto realIdx = idx;
            while (particles_.getGhost()(realIdx) != -1) realIdx = particles_.getGhost()(realIdx);
            particles_.getGhost()(nextParticle) = realIdx;
            ++particles_.numGhostParticles;
        }

        if (particles_.getPos(idx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto nextParticle = particles_.numLocalParticles + particles_.numGhostParticles;
            particles_.copy(idx, nextParticle);
            particles_.getPos(nextParticle, dim) += subdomain_.diameter[dim];
            auto realIdx = idx;
            while (particles_.getGhost()(realIdx) != -1) realIdx = particles_.getGhost()(realIdx);
            particles_.getGhost()(nextParticle) = realIdx;
            ++particles_.numGhostParticles;
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TagX, const idx_t& idx) const { copySelf(idx, 0); }
    KOKKOS_INLINE_FUNCTION
    void operator()(TagY, const idx_t& idx) const { copySelf(idx, 1); }
    KOKKOS_INLINE_FUNCTION
    void operator()(TagZ, const idx_t& idx) const { copySelf(idx, 2); }

    HaloExchange(Particles& particles, const Subdomain& subdomain)
        : particles_(particles), subdomain_(subdomain)
    {
    }
};
