#pragma once

#include "data/Atoms.hpp"
#include "data/EnergyAndVirialReducer.hpp"
#include "datatypes.hpp"

namespace mrmd::action::impl
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

    struct ForceAndEnergy
    {
        real_t forceFactor;
        real_t energy;
    };

private:
    Kokkos::View<PrecomputedValues*> precomputedValues_;
    bool isShifted_;  ///< potential is shifted at rc to 0

public:
    KOKKOS_INLINE_FUNCTION
    ForceAndEnergy computeForceAndEnergy(const real_t& distSqr, const idx_t& typeIdx) const
    {
        ForceAndEnergy ret;
        if (distSqr >= precomputedValues_(typeIdx).cappingDistanceSqr)
        {
            // normal LJ calculation
            auto frac2 = 1_r / distSqr;
            auto frac6 = frac2 * frac2 * frac2;
            ret.forceFactor =
                frac6 *
                (precomputedValues_(typeIdx).ff1 * frac6 - precomputedValues_(typeIdx).ff2) * frac2;
            ret.energy = frac6 * (precomputedValues_(typeIdx).ef1 * frac6 -
                                  precomputedValues_(typeIdx).ef2) -
                         precomputedValues_(typeIdx).shift;
            return ret;
        }

        // force capping
        auto dist = std::sqrt(distSqr);
        ret.forceFactor = precomputedValues_(typeIdx).cappingCoeff / dist;
        ret.energy = precomputedValues_(typeIdx).energyAtCappingPoint -
                     (dist - precomputedValues_(typeIdx).cappingDistance) *
                         precomputedValues_(typeIdx).cappingCoeff -
                     precomputedValues_(typeIdx).shift;
        return ret;
    }

    /**
     * Initialize shift and capping parameters.
     * Will be called at initialization.
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& typeIdx) const
    {
        // reset capping distance to calculate capping factors with real functions
        auto capDist = precomputedValues_(typeIdx).cappingDistance;
        precomputedValues_(typeIdx).cappingDistance = 0_r;
        precomputedValues_(typeIdx).cappingDistanceSqr = 0_r;
        auto forceAndEnergy = computeForceAndEnergy(capDist * capDist, typeIdx);
        precomputedValues_(typeIdx).cappingCoeff = forceAndEnergy.forceFactor * capDist;
        precomputedValues_(typeIdx).energyAtCappingPoint = forceAndEnergy.energy;
        precomputedValues_(typeIdx).cappingDistance = capDist;
        precomputedValues_(typeIdx).cappingDistanceSqr = capDist * capDist;

        if (isShifted_)
        {
            precomputedValues_(typeIdx).shift =
                computeForceAndEnergy(precomputedValues_(typeIdx).rcSqr, typeIdx).energy;
        }
    }

    CappedLennardJonesPotential(const std::vector<real_t>& cappingDistance,
                                const std::vector<real_t>& rc,
                                const std::vector<real_t>& sigma,
                                const std::vector<real_t>& epsilon,
                                const idx_t& numTypes,
                                const bool isShifted)
        : isShifted_(isShifted)
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
        Kokkos::parallel_for(policy, *this);
        Kokkos::fence();
    }
};
}  // namespace mrmd::action::impl

namespace mrmd::action
{
class LennardJones
{
private:
    impl::CappedLennardJonesPotential LJ_;
    real_t rcSqr_;
    data::Atoms::pos_t pos_;
    data::Atoms::force_t::atomic_access_slice force_;
    data::Atoms::type_t type_;

    VerletList verletList_;

    const idx_t numTypes_;

    data::EnergyAndVirialReducer energyAndVirial_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx, data::EnergyAndVirialReducer& energyAndVirial) const
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
            auto forceAndEnergy = LJ_.computeForceAndEnergy(distSqr, typeIdx);
            assert(!std::isnan(forceAndEnergy.forceFactor));
            energyAndVirial.energy += forceAndEnergy.energy;
            energyAndVirial.virial -= 0.5_r * forceAndEnergy.forceFactor * distSqr;

            forceTmp[0] += dx * forceAndEnergy.forceFactor;
            forceTmp[1] += dy * forceAndEnergy.forceFactor;
            forceTmp[2] += dz * forceAndEnergy.forceFactor;

            force_(jdx, 0) -= dx * forceAndEnergy.forceFactor;
            force_(jdx, 1) -= dy * forceAndEnergy.forceFactor;
            force_(jdx, 2) -= dz * forceAndEnergy.forceFactor;
        }

        force_(idx, 0) += forceTmp[0];
        force_(idx, 1) += forceTmp[1];
        force_(idx, 2) += forceTmp[2];
    }

    real_t getEnergy() const { return energyAndVirial_.energy; }
    real_t getVirial() const { return energyAndVirial_.virial; }

    void apply(data::Atoms& atoms, VerletList& verletList);

    LennardJones(const real_t rc,
                 const real_t& sigma,
                 const real_t& epsilon,
                 const real_t& cappingDistance = 0_r);

    LennardJones(const std::vector<real_t>& cappingDistance,
                 const std::vector<real_t>& rc,
                 const std::vector<real_t>& sigma,
                 const std::vector<real_t>& epsilon,
                 const idx_t& numTypes,
                 const bool isShifted);
};
}  // namespace mrmd::action