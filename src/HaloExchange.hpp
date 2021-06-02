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
            particles_.getPos(nextParticle, 0) = particles_.getPos(idx, 0);
            particles_.getPos(nextParticle, 1) = particles_.getPos(idx, 1);
            particles_.getPos(nextParticle, 2) = particles_.getPos(idx, 2);
            particles_.getPos(nextParticle, dim) -= subdomain_.diameter[dim];
            ++particles_.numGhostParticles;
        }

        if (particles_.getPos(idx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto nextParticle = particles_.numLocalParticles + particles_.numGhostParticles;
            particles_.getPos(nextParticle, 0) = particles_.getPos(idx, 0);
            particles_.getPos(nextParticle, 1) = particles_.getPos(idx, 1);
            particles_.getPos(nextParticle, 2) = particles_.getPos(idx, 2);
            particles_.getPos(nextParticle, dim) += subdomain_.diameter[dim];
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
        particles_.numGhostParticles = 0;
    }
};
