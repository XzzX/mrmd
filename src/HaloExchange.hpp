#pragma once

#include "Particles.hpp"
#include "Subdomain.hpp"

class HaloExchange
{
private:
    Particles& particles_;
    const Subdomain& subdomain_;
    mutable idx_t nextParticle_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            if (particles_.getPos(idx, dim) > subdomain_.maxInnerCorner[dim])
            {
                particles_.getPos(nextParticle_, 0) = particles_.getPos(idx, 0);
                particles_.getPos(nextParticle_, 1) = particles_.getPos(idx, 1);
                particles_.getPos(nextParticle_, 2) = particles_.getPos(idx, 2);
                particles_.getPos(idx, dim) -= subdomain_.diameter[dim];
                ++nextParticle_;
            }

            if (particles_.getPos(idx, dim) < subdomain_.minInnerCorner[dim])
            {
                particles_.getPos(nextParticle_, 0) = particles_.getPos(idx, 0);
                particles_.getPos(nextParticle_, 1) = particles_.getPos(idx, 1);
                particles_.getPos(nextParticle_, 2) = particles_.getPos(idx, 2);
                particles_.getPos(idx, dim) += subdomain_.diameter[dim];
                ++nextParticle_;
            }
        }
    }

    HaloExchange(Particles& particles, const Subdomain& subdomain)
        : particles_(particles), subdomain_(subdomain)
    {
        nextParticle_ = particles_.numLocalParticles;
    }

    auto getNumParticles(){return nextParticle_;}
};
