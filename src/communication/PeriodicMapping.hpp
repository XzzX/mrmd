#pragma once

#include "checks.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

class PeriodicMapping
{
private:
    Particles::pos_t pos_;
    const Subdomain subdomain_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        for (auto dim = 0; dim < Particles::DIMENSIONS; ++dim)
        {
            auto& x = pos_(idx, dim);
            if (x > subdomain_.maxCorner[dim]) x -= subdomain_.diameter[dim];
            if (x < subdomain_.minCorner[dim]) x += subdomain_.diameter[dim];
            CHECK_LESS_EQUAL(x, subdomain_.maxCorner[dim]);
            CHECK_GREATER_EQUAL(x, subdomain_.minCorner[dim]);
        }
    }

    void mapIntoDomain(Particles& particles)
    {
        pos_ = particles.getPos();
        auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
        Kokkos::parallel_for(policy, *this);
    }

    PeriodicMapping(const Subdomain& subdomain) : subdomain_(subdomain) {}
};
