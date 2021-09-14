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
private:
    std::vector<real_t> ff1_;  ///< force factor 1
    std::vector<real_t> ff2_;  ///< force factor 2
    std::vector<real_t> ef1_;  ///< energy factor 1
    std::vector<real_t> ef2_;  ///< energy factor 2
    std::vector<real_t> rcSqr_;
    std::vector<real_t> cappingDistance_;
    std::vector<real_t> cappingDistanceSqr_;
    std::vector<real_t> cappingCoeff_;
    std::vector<real_t> shift_;  ///< shifting the potential at rc to zero
    std::vector<real_t> energyAtCappingPoint_;

public:
    const auto& getShift() const { return shift_; }

    KOKKOS_INLINE_FUNCTION
    real_t computeForce(const real_t& distSqr, const idx_t& typeIdx) const
    {
        if (distSqr >= cappingDistanceSqr_[typeIdx])
        {
            // normal LJ force calculation
            auto frac2 = 1_r / distSqr;
            auto frac6 = frac2 * frac2 * frac2;
            return frac6 * (ff1_[typeIdx] * frac6 - ff2_[typeIdx]) * frac2;
        }

        // force capping
        return cappingCoeff_[typeIdx] / std::sqrt(distSqr);
    }

    KOKKOS_INLINE_FUNCTION
    real_t computeEnergy(const real_t& distSqr, const idx_t& typeIdx) const
    {
        if (distSqr >= cappingDistanceSqr_[typeIdx])
        {
            // normal LJ energy calculation
            real_t frac2 = 1_r / distSqr;
            real_t frac6 = frac2 * frac2 * frac2;
            return frac6 * (ef1_[typeIdx] * frac6 - ef2_[typeIdx]) - shift_[typeIdx];
        }

        // capped energy
        return energyAtCappingPoint_[typeIdx] -
               (std::sqrt(distSqr) - cappingDistance_[typeIdx]) * cappingCoeff_[typeIdx] -
               shift_[typeIdx];
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

        ff1_.resize(numTypes * numTypes);
        ff2_.resize(numTypes * numTypes);
        ef1_.resize(numTypes * numTypes);
        ef2_.resize(numTypes * numTypes);
        rcSqr_.resize(numTypes * numTypes);
        cappingDistance_.resize(numTypes * numTypes);
        cappingDistanceSqr_.resize(numTypes * numTypes);
        cappingCoeff_.resize(numTypes * numTypes);
        shift_.resize(numTypes * numTypes);
        energyAtCappingPoint_.resize(numTypes * numTypes);

        for (idx_t typeIdx = 0; typeIdx < numTypes * numTypes; ++typeIdx)
        {
            auto sig2 = sigma[typeIdx] * sigma[typeIdx];
            auto sig6 = sig2 * sig2 * sig2;
            ff1_[typeIdx] = 48_r * epsilon[typeIdx] * sig6 * sig6;
            ff2_[typeIdx] = 24_r * epsilon[typeIdx] * sig6;
            ef1_[typeIdx] = 4_r * epsilon[typeIdx] * sig6 * sig6;
            ef2_[typeIdx] = 4_r * epsilon[typeIdx] * sig6;

            rcSqr_[typeIdx] = rc[typeIdx] * rc[typeIdx];

            // parameters for the capped part of LJ, use uncapped LJ to compute cappingCoeff
            cappingDistance_[typeIdx] = 0_r;
            cappingDistanceSqr_[typeIdx] = 0_r;
            cappingCoeff_[typeIdx] =
                computeForce(cappingDistance[typeIdx] * cappingDistance[typeIdx], typeIdx) *
                cappingDistance[typeIdx];
            energyAtCappingPoint_[typeIdx] =
                computeEnergy(cappingDistance[typeIdx] * cappingDistance[typeIdx], typeIdx);
            cappingDistance_[typeIdx] = cappingDistance[typeIdx];
            cappingDistanceSqr_[typeIdx] = cappingDistance[typeIdx] * cappingDistance[typeIdx];

            if (doShift)
            {
                shift_[typeIdx] = computeEnergy(rcSqr_[typeIdx], typeIdx);
            }
        }

        ff1_.shrink_to_fit();
        ff2_.shrink_to_fit();
        ef1_.shrink_to_fit();
        ef2_.shrink_to_fit();
        rcSqr_.shrink_to_fit();
        cappingDistance_.shrink_to_fit();
        cappingDistanceSqr_.shrink_to_fit();
        cappingCoeff_.shrink_to_fit();
        shift_.shrink_to_fit();
        energyAtCappingPoint_.shrink_to_fit();
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