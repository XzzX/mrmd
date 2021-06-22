#pragma once

#include "checks.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace impl
{
class HaloExchange
{
private:
    Particles& particles_;
    const Subdomain& subdomain_;

    Particles::pos_t pos_;
    Particles::ghost_t ghost_;

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
        if (pos_(idx, dim) > subdomain_.maxInnerCorner[dim])
        {
            auto nextParticle = particles_.numLocalParticles +
                                Kokkos::atomic_fetch_add(&particles_.numGhostParticles, 1);
            particles_.copy(nextParticle, idx);
            pos_(nextParticle, dim) -= subdomain_.diameter[dim];
            CHECK_LESS(pos_(nextParticle, dim), subdomain_.minCorner[dim]);
            auto realIdx = idx;
            while (ghost_(realIdx) != -1) realIdx = ghost_(realIdx);
            ghost_(nextParticle) = realIdx;
            CHECK_NOT_EQUAL(ghost_(nextParticle), -1);
            CHECK_GREATER_EQUAL(ghost_(nextParticle), 0);
            CHECK_LESS(ghost_(nextParticle), particles_.numLocalParticles);
        }

        if (pos_(idx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto nextParticle = particles_.numLocalParticles +
                                Kokkos::atomic_fetch_add(&particles_.numGhostParticles, 1);
            particles_.copy(nextParticle, idx);
            pos_(nextParticle, dim) += subdomain_.diameter[dim];
            CHECK_GREATER(pos_(nextParticle, dim), subdomain_.maxCorner[dim]);
            auto realIdx = idx;
            while (ghost_(realIdx) != -1) realIdx = ghost_(realIdx);
            ghost_(nextParticle) = realIdx;
            CHECK_NOT_EQUAL(ghost_(nextParticle), -1);
            CHECK_GREATER_EQUAL(ghost_(nextParticle), 0);
            CHECK_LESS(ghost_(nextParticle), particles_.numLocalParticles);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(TagX, const idx_t& idx) const { copySelf(idx, 0); }
    KOKKOS_INLINE_FUNCTION
    void operator()(TagY, const idx_t& idx) const { copySelf(idx, 1); }
    KOKKOS_INLINE_FUNCTION
    void operator()(TagZ, const idx_t& idx) const { copySelf(idx, 2); }

    HaloExchange(const Subdomain& subdomain, Particles& particles)
        : subdomain_(subdomain), particles_(particles)
    {
        pos_ = particles.getPos();
        ghost_ = particles.getGhost();
    }
};
}  // namespace impl

class HaloExchange
{
private:
    const Subdomain& subdomain_;

public:
    void createGhostsXYZ(Particles& particles)
    {
        impl::HaloExchange haloExchange(subdomain_, particles);

        auto policyX = Kokkos::RangePolicy<impl::HaloExchange::TagX>(
            0, particles.numLocalParticles + particles.numGhostParticles);
        Kokkos::parallel_for(policyX, haloExchange);
        Kokkos::fence();

        auto policyY = Kokkos::RangePolicy<impl::HaloExchange::TagY>(
            0, particles.numLocalParticles + particles.numGhostParticles);
        Kokkos::parallel_for(policyY, haloExchange);
        Kokkos::fence();

        auto policyZ = Kokkos::RangePolicy<impl::HaloExchange::TagZ>(
            0, particles.numLocalParticles + particles.numGhostParticles);
        Kokkos::parallel_for(policyZ, haloExchange);
        Kokkos::fence();
    }

    HaloExchange(const Subdomain& subdomain) : subdomain_(subdomain) {}
};
