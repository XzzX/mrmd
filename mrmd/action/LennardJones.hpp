// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    KOKKOS_FUNCTION
    ForceAndEnergy computeForceAndEnergy(const real_t& distSqr, const idx_t& typeIdx) const;

    /**
     * Initialize shift and capping parameters.
     * Will be called at initialization.
     */
    KOKKOS_FUNCTION
    void operator()(const idx_t& typeIdx) const;

    CappedLennardJonesPotential(const std::vector<real_t>& cappingDistance,
                                const std::vector<real_t>& rc,
                                const std::vector<real_t>& sigma,
                                const std::vector<real_t>& epsilon,
                                const idx_t& numTypes,
                                const bool isShifted);
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

    HalfVerletList verletList_;

    const idx_t numTypes_;

    data::EnergyAndVirialReducer energyAndVirial_;

public:
    KOKKOS_FUNCTION
    void operator()(const idx_t& idx, data::EnergyAndVirialReducer& energyAndVirial) const;

    real_t getEnergy() const;
    real_t getVirial() const;

    void apply(data::Atoms& atoms, HalfVerletList& verletList);

    template <std::predicate<const real_t,
                             const real_t,
                             const real_t,
                             const real_t,
                             const real_t,
                             const real_t> BinaryPred>
    void apply_if(const data::Atoms& atoms,
                  const HalfVerletList& verletList,
                  const BinaryPred& pred);

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

template <std::predicate<const real_t,
                         const real_t,
                         const real_t,
                         const real_t,
                         const real_t,
                         const real_t> BinaryPred>
void LennardJones::apply_if(const data::Atoms& atoms,
                            const HalfVerletList& verletList,
                            const BinaryPred& pred)
{
    energyAndVirial_ = data::EnergyAndVirialReducer();
    pos_ = atoms.getPos();
    force_ = atoms.getForce();
    type_ = atoms.getType();
    verletList_ = verletList;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, data::EnergyAndVirialReducer& energyAndVirial)
    {
        real_t posTmp[3];
        posTmp[0] = pos_(idx, 0);
        posTmp[1] = pos_(idx, 1);
        posTmp[2] = pos_(idx, 2);

        real_t forceTmp[3] = {0_r, 0_r, 0_r};

        const auto numNeighbors = idx_c(HalfNeighborList::numNeighbor(verletList_, idx));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            idx_t jdx = idx_c(HalfNeighborList::getNeighbor(verletList_, idx, n));
            assert(0 <= jdx);

            if (!pred(posTmp[0], posTmp[1], posTmp[2], pos_(jdx, 0), pos_(jdx, 1), pos_(jdx, 2)))
                continue;

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
    };
    Kokkos::parallel_reduce("LennardJones::apply_if", policy, kernel, energyAndVirial_);
    Kokkos::fence();
}

}  // namespace mrmd::action