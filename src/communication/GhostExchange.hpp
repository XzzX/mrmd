#pragma once

#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class GhostExchange
{
private:
    data::Particles particles_ = data::Particles(0);
    const data::Subdomain subdomain_;

    data::Particles::pos_t pos_;

    Kokkos::View<idx_t> newGhostCounter_;
    Kokkos::View<idx_t>::host_mirror_type hNewGhostCounter_;
    /// Stores the corresponding real particle index for every ghost particle.
    IndexView correspondingRealParticle_;

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
    idx_t findRealIdx(const idx_t src) const
    {
        auto realIdx = src;
        while (correspondingRealParticle_(realIdx) != -1)
        {
            realIdx = correspondingRealParticle_(realIdx);
            assert(0 <= realIdx);
            assert(realIdx < particles_.numLocalParticles + particles_.numGhostParticles);
        }
        return realIdx;
    }

    KOKKOS_INLINE_FUNCTION
    void copySelf(const idx_t idx, const idx_t dim) const
    {
        if (pos_(idx, dim) > subdomain_.maxInnerCorner[dim])
        {
            auto newGhostIdx = particles_.numLocalParticles + particles_.numGhostParticles +
                               Kokkos::atomic_fetch_add(&newGhostCounter_(), 1);
            if (newGhostIdx < particles_.size())
            {
                particles_.copy(newGhostIdx, idx);
                pos_(newGhostIdx, dim) -= subdomain_.diameter[dim];
                assert(pos_(newGhostIdx, dim) < subdomain_.minCorner[dim]);
                assert(pos_(newGhostIdx, dim) > subdomain_.minGhostCorner[dim]);
                correspondingRealParticle_(newGhostIdx) = findRealIdx(idx);
            }
        }

        if (pos_(idx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto newGhostIdx = particles_.numLocalParticles + particles_.numGhostParticles +
                               Kokkos::atomic_fetch_add(&newGhostCounter_(), 1);
            if (newGhostIdx < particles_.size())
            {
                particles_.copy(newGhostIdx, idx);
                pos_(newGhostIdx, dim) += subdomain_.diameter[dim];
                assert(pos_(newGhostIdx, dim) > subdomain_.maxCorner[dim]);
                assert(pos_(newGhostIdx, dim) < subdomain_.maxGhostCorner[dim]);
                correspondingRealParticle_(newGhostIdx) = findRealIdx(idx);
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_X, const idx_t& idx) const { copySelf(idx, 0); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_Y, const idx_t& idx) const { copySelf(idx, 1); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_Z, const idx_t& idx) const { copySelf(idx, 2); }

    template <typename EXCHANGE_DIRECTION>
    IndexView exchangeGhosts(data::Particles& particles)
    {
        if (correspondingRealParticle_.extent(0) < particles.numLocalParticles)
        {
            // initialize correspondingRealParticle_ for all real particles
            Kokkos::resize(correspondingRealParticle_, particles.numLocalParticles);
            Kokkos::deep_copy(correspondingRealParticle_, -1);
        }
        assert(correspondingRealParticle_.extent(0) >= particles.size());

        auto newSize = particles.numLocalParticles + particles.numGhostParticles;
        do
        {
            if (newSize > particles.size())
            {
                // resize
                particles.resize(newSize);
                if (correspondingRealParticle_.extent(0) < newSize)
                {
                    Kokkos::resize(correspondingRealParticle_, newSize);
                }
            }

            particles_ = particles;
            pos_ = particles.getPos();

            Kokkos::deep_copy(newGhostCounter_, 0);

            auto policy = Kokkos::RangePolicy<EXCHANGE_DIRECTION>(
                0, particles.numLocalParticles + particles.numGhostParticles);
            Kokkos::parallel_for(policy, *this, "GhostExchange::exchangeGhosts");
            Kokkos::fence();

            Kokkos::deep_copy(hNewGhostCounter_, newGhostCounter_);
            newSize =
                particles.numLocalParticles + particles.numGhostParticles + hNewGhostCounter_();
        } while (newSize > particles.size());  // resize and rerun

        particles.numGhostParticles += hNewGhostCounter_();
        return correspondingRealParticle_;
    }

    IndexView createGhostParticlesXYZ(data::Particles& particles)
    {
        particles.numGhostParticles = 0;
        if (correspondingRealParticle_.extent(0) < particles.size())
            Kokkos::resize(correspondingRealParticle_, particles.size());
        Kokkos::deep_copy(correspondingRealParticle_, -1);

        exchangeGhosts<impl::GhostExchange::DIRECTION_X>(particles);
        exchangeGhosts<impl::GhostExchange::DIRECTION_Y>(particles);
        exchangeGhosts<impl::GhostExchange::DIRECTION_Z>(particles);
        Kokkos::fence();

        particles.resize(particles.numLocalParticles + particles.numGhostParticles);
        return correspondingRealParticle_;
    }

    GhostExchange(const data::Subdomain& subdomain)
        : subdomain_(subdomain),
          newGhostCounter_("newGhostCounter"),
          hNewGhostCounter_("hNewGhostCounter"),
          correspondingRealParticle_("correspondingRealParticle", 0)
    {
    }
};
}  // namespace impl
}  // namespace communication
}  // namespace mrmd
