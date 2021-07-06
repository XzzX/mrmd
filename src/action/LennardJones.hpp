#pragma once

#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class LennardJones
{
private:
    const real_t epsilon_;
    real_t sig2_;
    real_t sig6_;
    real_t ff1_;
    real_t ff2_;
    real_t rcSqr_;
    data::Particles::pos_t pos_;
    data::Particles::force_t::atomic_access_slice force_;

    VerletList verletList_;

public:
    KOKKOS_INLINE_FUNCTION
    real_t computeForce_(const real_t& distSqr) const
    {
        auto frac2 = 1.0 / distSqr;
        auto frac6 = frac2 * frac2 * frac2;
        return frac6 * (ff1_ * frac6 - ff2_) * frac2;
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        real_t posTmp[3];
        posTmp[0] = pos_(idx, 0);
        posTmp[1] = pos_(idx, 1);
        posTmp[2] = pos_(idx, 2);

        real_t forceTmp[3] = {0_r, 0_r, 0_r};

        const auto numNeighbors = idx_c(NeighborList::numNeighbor(verletList_, idx));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            idx_t jdx = idx_c(NeighborList::getNeighbor(verletList_, idx, n));
            assert(0 <= jdx);
            assert(jdx < 60000);

            auto dx = posTmp[0] - pos_(jdx, 0);
            auto dy = posTmp[1] - pos_(jdx, 1);
            auto dz = posTmp[2] - pos_(jdx, 2);

            auto distSqr = dx * dx + dy * dy + dz * dz;

            if (distSqr > rcSqr_) continue;

            auto ffactor = computeForce_(distSqr);

            forceTmp[0] += dx * ffactor;
            forceTmp[1] += dy * ffactor;
            forceTmp[2] += dz * ffactor;

            force_(jdx, 0) -= dx * ffactor;
            force_(jdx, 1) -= dy * ffactor;
            force_(jdx, 2) -= dz * ffactor;
        }

        force_(idx, 0) += forceTmp[0];
        force_(idx, 1) += forceTmp[1];
        force_(idx, 2) += forceTmp[2];
    }

    KOKKOS_INLINE_FUNCTION
    real_t computeEnergy_(const real_t& distSqr) const
    {
        real_t frac2 = sig2_ / distSqr;
        real_t frac6 = frac2 * frac2 * frac2;
        return 4.0 * epsilon_ * (frac6 * frac6 - frac6);
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx, const idx_t& jdx, real_t& energy) const
    {
        auto dx = pos_(idx, 0) - pos_(jdx, 0);
        auto dy = pos_(idx, 1) - pos_(jdx, 1);
        auto dz = pos_(idx, 2) - pos_(jdx, 2);
        auto distSqr = dx * dx + dy * dy + dz * dz;

        if (distSqr > rcSqr_) return;

        energy += computeEnergy_(distSqr);
    }

    void applyForces(data::Particles& particles, VerletList& verletList)
    {
        pos_ = particles.getPos();
        force_ = particles.getForce();
        verletList_ = verletList;

        auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
        Kokkos::parallel_for(policy, *this, "LennardJones::applyForces");

        Kokkos::fence();
    }

    template <typename VERLET_LIST>
    real_t computeEnergy(data::Particles& particles, VERLET_LIST& verletList)
    {
        pos_ = particles.getPos();

        real_t E0 = 0_r;
        Cabana::neighbor_parallel_reduce(Kokkos::RangePolicy<>(0, particles.numLocalParticles),
                                         *this,
                                         verletList,
                                         Cabana::FirstNeighborsTag(),
                                         Cabana::TeamOpTag(),
                                         E0,
                                         "LennardJones::computeEnergy");

        return E0;
    }

    LennardJones(const real_t rc, const real_t& sigma, const real_t& epsilon)
        : epsilon_(epsilon), rcSqr_(rc * rc)
    {
        sig2_ = sigma * sigma;
        sig6_ = sig2_ * sig2_ * sig2_;
        ff1_ = 48.0 * epsilon * sig6_ * sig6_;
        ff2_ = 24.0 * epsilon * sig6_;
    }
};
}  // namespace action
}  // namespace mrmd