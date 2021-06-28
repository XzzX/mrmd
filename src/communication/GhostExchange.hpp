#pragma once

#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace communication
{
namespace impl
{
class GhostExchange
{
private:
    Particles particles_ = Particles(0);
    const Subdomain subdomain_;

    Particles::pos_t pos_;
    Particles::ghost_t ghost_;

    Kokkos::View<idx_t> newGhostCounter_;

public:
    struct DIRECTION_X
    {
    };
    struct DIRECTION_Y
    {
    };
    struct DIRECTION_Z
    {
    };

    KOKKOS_INLINE_FUNCTION
    void copySelf(const idx_t idx, const idx_t dim) const
    {
        if (pos_(idx, dim) > subdomain_.maxInnerCorner[dim])
        {
            auto nextParticle = particles_.numLocalParticles + particles_.numGhostParticles +
                                Kokkos::atomic_fetch_add(&newGhostCounter_(), 1);
            particles_.copyAsGhost(nextParticle, idx);
            pos_(nextParticle, dim) -= subdomain_.diameter[dim];
            assert(pos_(nextParticle, dim) < subdomain_.minCorner[dim]);
            assert(pos_(nextParticle, dim) > subdomain_.minGhostCorner[dim]);
        }

        if (pos_(idx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto nextParticle = particles_.numLocalParticles + particles_.numGhostParticles +
                                Kokkos::atomic_fetch_add(&newGhostCounter_(), 1);
            particles_.copyAsGhost(nextParticle, idx);
            pos_(nextParticle, dim) += subdomain_.diameter[dim];
            assert(pos_(nextParticle, dim) > subdomain_.maxCorner[dim]);
            assert(pos_(nextParticle, dim) < subdomain_.maxGhostCorner[dim]);
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_X, const idx_t& idx) const { copySelf(idx, 0); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_Y, const idx_t& idx) const { copySelf(idx, 1); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_Z, const idx_t& idx) const { copySelf(idx, 2); }

    template <typename EXCHANGE_DIRECTION>
    void exchangeGhosts(Particles& particles)
    {
        particles_ = particles;
        pos_ = particles.getPos();
        ghost_ = particles.getGhost();

        Kokkos::deep_copy(newGhostCounter_, 0);
        auto policy = Kokkos::RangePolicy<EXCHANGE_DIRECTION>(
            0, particles_.numLocalParticles + particles_.numGhostParticles);
        Kokkos::parallel_for(policy, *this, "GhostExchange::exchangeGhosts");
        Kokkos::fence();
        auto hNewGhostCounter =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), newGhostCounter_);
        particles.numGhostParticles += hNewGhostCounter();
    }

    void createGhostParticlesXYZ(Particles& particles)
    {
        particles.resize(100000);
        particles.removeGhostParticles();

        impl::GhostExchange ghostExchange(subdomain_);
        ghostExchange.exchangeGhosts<impl::GhostExchange::DIRECTION_X>(particles);
        ghostExchange.exchangeGhosts<impl::GhostExchange::DIRECTION_Y>(particles);
        ghostExchange.exchangeGhosts<impl::GhostExchange::DIRECTION_Z>(particles);
        Kokkos::fence();

        particles.resize(particles.numLocalParticles + particles.numGhostParticles);
    }

    GhostExchange(const Subdomain& subdomain)
        : subdomain_(subdomain), newGhostCounter_("newGhostCounter")
    {
    }
};
}  // namespace impl
}  // namespace communication
