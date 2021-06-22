#pragma once

#include "checks.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

class PeriodicMapping
{
private:
    const Subdomain& subdomain_;

public:
    void mapIntoDomain(Particles& particles)
    {
        auto pos = particles.getPos();
        auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
        auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
        {
            for (auto dim = 0; dim < Particles::dim; ++dim)
            {
                auto& x = pos(idx, dim);
                if (x > subdomain_.maxCorner[dim]) x -= subdomain_.diameter[dim];
                if (x < subdomain_.minCorner[dim]) x += subdomain_.diameter[dim];
                CHECK_LESS_EQUAL(x, subdomain_.maxCorner[dim]);
                CHECK_GREATER_EQUAL(x, subdomain_.minCorner[dim]);
            }
        };
        Kokkos::parallel_for(policy, kernel);
    }

    PeriodicMapping(const Subdomain& subdomain) : subdomain_(subdomain) {}
};
