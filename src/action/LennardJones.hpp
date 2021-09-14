#pragma once

#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
namespace impl
{
class CappedLennardJonesPotential
{
public:
    struct PrecomputedValues
    {
        real_t ff1;  ///< force factor 1
        real_t ff2;  ///< force factor 2
        real_t ef1;  ///< energy factor 1
        real_t ef2;  ///< energy factor 2
        real_t rcSqr;
        real_t cappingDistance;
        real_t cappingDistanceSqr;
        real_t cappingCoeff;
        real_t shift;  ///< shifting the potential at rc to zero
        real_t energyAtCappingPoint;
    };

private:
    Kokkos::View<PrecomputedValues*> precomputedValues_;

public:
    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& distSqr, const idx_t& typeIdx) const
    {
        if (distSqr >= precomputedValues_(typeIdx).cappingDistanceSqr)
        {
            // normal LJ force calculation
            auto frac2 = 1_r / distSqr;
            auto frac6 = frac2 * frac2 * frac2;
            return frac6 *
                   (precomputedValues_(typeIdx).ff1 * frac6 - precomputedValues_(typeIdx).ff2) *
                   frac2;
        }

        // force capping
        return precomputedValues_(typeIdx).cappingCoeff / std::sqrt(distSqr);
    }

    KOKKOS_INLINE_FUNCTION
    real_t computeEnergy(const real_t& distSqr, const idx_t& typeIdx) const
    {
        if (distSqr >= precomputedValues_(typeIdx).cappingDistanceSqr)
        {
            // normal LJ energy calculation
            real_t frac2 = 1_r / distSqr;
            real_t frac6 = frac2 * frac2 * frac2;
            return frac6 *
                       (precomputedValues_(typeIdx).ef1 * frac6 - precomputedValues_(typeIdx).ef2) -
                   precomputedValues_(typeIdx).shift;
        }

        // capped energy
        return precomputedValues_(typeIdx).energyAtCappingPoint -
               (std::sqrt(distSqr) - precomputedValues_(typeIdx).cappingDistance) *
                   precomputedValues_(typeIdx).cappingCoeff -
               precomputedValues_(typeIdx).shift;
    }

    CappedLennardJonesPotential(const std::vector<real_t>& cappingDistance,
                                const std::vector<real_t>& rc,
                                const std::vector<real_t>& sigma,
                                const std::vector<real_t>& epsilon,
                                const idx_t& numTypes,
                                const bool doShift)
    {
        assert(cappingDistance.size() == numTypes * numTypes);
        assert(rc.size() == numTypes * numTypes);
        assert(sigma.size() == numTypes * numTypes);
        assert(epsilon.size() == numTypes * numTypes);

        precomputedValues_ = Kokkos::View<PrecomputedValues*>(
            "CappedLennardJonesPotential::PrecomputedValues", numTypes * numTypes);
        auto hPrecomputedValues =
            Kokkos::create_mirror_view(Kokkos::HostSpace(), precomputedValues_);

        for (idx_t typeIdx = 0; typeIdx < numTypes * numTypes; ++typeIdx)
        {
            auto sig2 = sigma[typeIdx] * sigma[typeIdx];
            auto sig6 = sig2 * sig2 * sig2;
            hPrecomputedValues(typeIdx).ff1 = 48_r * epsilon[typeIdx] * sig6 * sig6;
            hPrecomputedValues(typeIdx).ff2 = 24_r * epsilon[typeIdx] * sig6;
            hPrecomputedValues(typeIdx).ef1 = 4_r * epsilon[typeIdx] * sig6 * sig6;
            hPrecomputedValues(typeIdx).ef2 = 4_r * epsilon[typeIdx] * sig6;

            hPrecomputedValues(typeIdx).rcSqr = rc[typeIdx] * rc[typeIdx];

            hPrecomputedValues(typeIdx).cappingDistance = cappingDistance[typeIdx];
        }
        Kokkos::deep_copy(precomputedValues_, hPrecomputedValues);

        auto policy = Kokkos::RangePolicy<>(0, numTypes * numTypes);
        auto kernel = [*this, doShift](const idx_t typeIdx)
        {
            // reset capping distance to calculate capping factors with real functions
            auto capDist = precomputedValues_(typeIdx).cappingDistance;
            precomputedValues_(typeIdx).cappingDistance = 0_r;
            precomputedValues_(typeIdx).cappingDistanceSqr = 0_r;
            precomputedValues_(typeIdx).cappingCoeff =
                computeForce(capDist * capDist, typeIdx) * capDist;
            precomputedValues_(typeIdx).energyAtCappingPoint =
                computeEnergy(capDist * capDist, typeIdx);
            precomputedValues_(typeIdx).cappingDistance = capDist;
            precomputedValues_(typeIdx).cappingDistanceSqr = capDist * capDist;

            if (doShift)
            {
                precomputedValues_(typeIdx).shift =
                    computeEnergy(precomputedValues_(typeIdx).rcSqr, typeIdx);
            }
        };
        Kokkos::parallel_for(policy, kernel);
        Kokkos::fence();
    }
};
}  // namespace impl

class LennardJones
{
private:
    impl::CappedLennardJonesPotential LJ_;
    real_t rcSqr_;
    data::Particles::pos_t pos_;
    data::Particles::force_t::atomic_access_slice force_;
    data::Particles::type_t type_;

    VerletList verletList_;

    const idx_t numTypes_;

public:
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

            auto dx = posTmp[0] - pos_(jdx, 0);
            auto dy = posTmp[1] - pos_(jdx, 1);
            auto dz = posTmp[2] - pos_(jdx, 2);

            auto distSqr = dx * dx + dy * dy + dz * dz;

            if (distSqr > rcSqr_) continue;

            auto typeIdx = type_(idx) * numTypes_ + type_(jdx);
            auto ffactor = LJ_.computeForce(distSqr, typeIdx);

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
    void operator()(const idx_t& idx, const idx_t& jdx, real_t& energy) const
    {
        auto dx = pos_(idx, 0) - pos_(jdx, 0);
        auto dy = pos_(idx, 1) - pos_(jdx, 1);
        auto dz = pos_(idx, 2) - pos_(jdx, 2);
        auto distSqr = dx * dx + dy * dy + dz * dz;

        if (distSqr > rcSqr_) return;

        auto typeIdx = type_(idx) * numTypes_ + type_(jdx);
        energy += LJ_.computeEnergy(distSqr, typeIdx);
    }

    void applyForces(data::Particles& particles, VerletList& verletList)
    {
        pos_ = particles.getPos();
        force_ = particles.getForce();
        type_ = particles.getType();
        verletList_ = verletList;

        auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
        Kokkos::parallel_for(policy, *this, "LennardJones::applyForces");

        Kokkos::fence();
    }

    template <typename VERLET_LIST>
    real_t computeEnergy(data::Particles& particles, VERLET_LIST& verletList)
    {
        pos_ = particles.getPos();
        type_ = particles.getType();

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

    LennardJones(const real_t rc,
                 const real_t& sigma,
                 const real_t& epsilon,
                 const real_t& cappingDistance = 0_r)
        : LJ_({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, false), rcSqr_(rc * rc), numTypes_(1)
    {
    }
};
}  // namespace action
}  // namespace mrmd