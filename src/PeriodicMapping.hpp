#pragma once

#include "Particles.hpp"
#include "Subdomain.hpp"

#include "checks.hpp"

class PeriodicMapping
{
private:
    Particles& particles_;
    const Subdomain& subdomain_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        for (auto dim = 0; dim < Particles::dim; ++dim)
        {
            auto& x = particles_.getPos(idx, dim);
            if (x > subdomain_.maxCorner[dim]) x -= subdomain_.diameter[dim];
            if (x < subdomain_.minCorner[dim]) x += subdomain_.diameter[dim];
            CHECK_LESS_EQUAL(x, subdomain_.maxCorner[dim]);
            CHECK_GREATER_EQUAL(x, subdomain_.minCorner[dim]);
        }
    }

    PeriodicMapping(Particles& particles, const Subdomain& subdomain)
        : particles_(particles), subdomain_(subdomain)
    {
    }
};
