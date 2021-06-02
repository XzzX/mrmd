#pragma once

#include "Particles.hpp"
#include "Subdomain.hpp"

class HaloExchange
{
private:
    Particles& particles_;
    const Subdomain& subdomain_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            if (particles_.getPos(idx, dim) > subdomain_.maxInnerCorner[dim])
            {
                auto nextParticle = particles_.numLocalParticles + particles_.numGhostParticles;
                particles_.getPos(nextParticle, 0) = particles_.getPos(idx, 0);
                particles_.getPos(nextParticle, 1) = particles_.getPos(idx, 1);
                particles_.getPos(nextParticle, 2) = particles_.getPos(idx, 2);
                particles_.getPos(idx, dim) -= subdomain_.diameter[dim];
                ++particles_.numGhostParticles;
            }

            if (particles_.getPos(idx, dim) < subdomain_.minInnerCorner[dim])
            {
                auto nextParticle = particles_.numLocalParticles + particles_.numGhostParticles;
                particles_.getPos(nextParticle, 0) = particles_.getPos(idx, 0);
                particles_.getPos(nextParticle, 1) = particles_.getPos(idx, 1);
                particles_.getPos(nextParticle, 2) = particles_.getPos(idx, 2);
                particles_.getPos(idx, dim) += subdomain_.diameter[dim];
                ++particles_.numGhostParticles;
            }
        }
    }

    HaloExchange(Particles& particles, const Subdomain& subdomain)
        : particles_(particles), subdomain_(subdomain)
    {
        particles_.numGhostParticles = 0;
    }
};
