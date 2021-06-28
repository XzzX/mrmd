#pragma once

#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace communication
{
namespace impl
{
class PeriodicMapping
{
private:
    Particles::pos_t pos_;
    const Subdomain subdomain_;

public:
    /**
     *
     * @pre Particle position is at most one periodic copy away
     * from the subdomain.
     * @post Particle position lies within half-open interval
     * [min, max) for all coordinate dimensions.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        for (auto dim = 0; dim < Particles::DIMENSIONS; ++dim)
        {
            auto& x = pos_(idx, dim);
            if (subdomain_.maxCorner[dim] <= x)
            {
                x -= subdomain_.diameter[dim];
                if (x < subdomain_.minCorner[dim])
                {
                    x = subdomain_.minCorner[dim];
                }
            }
            if (x < subdomain_.minCorner[dim])
            {
                x += subdomain_.diameter[dim];
                if (subdomain_.maxCorner[dim] <= x)
                {
                    x = subdomain_.minCorner[dim];
                }
            }
            assert(x < subdomain_.maxCorner[dim]);
            assert(subdomain_.minCorner[dim] <= subdomain_.minCorner[dim]);
        }
    }

    void mapIntoDomain(Particles& particles)
    {
        pos_ = particles.getPos();
        auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
        Kokkos::parallel_for(policy, *this, "PeriodicMapping::mapIntoDomain");
    }

    PeriodicMapping(const Subdomain& subdomain) : subdomain_(subdomain) {}
};

}  // namespace impl
}  // namespace communication