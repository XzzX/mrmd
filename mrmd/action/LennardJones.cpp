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

#include "LennardJones.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace mrmd::action
{
void LennardJones::apply(data::Atoms& atoms, HalfVerletList& verletList)
{
    apply_if(
        atoms,
        verletList,
        KOKKOS_LAMBDA(
            const real_t, const real_t, const real_t, const real_t, const real_t, const real_t) {
            return true;
        });
}

template <std::predicate<const real_t,
                         const real_t,
                         const real_t,
                         const real_t,
                         const real_t,
                         const real_t> BinaryPred>
void LennardJones::apply_if(data::Atoms& atoms, HalfVerletList& verletList, const BinaryPred& pred)
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

KOKKOS_FUNCTION
void LennardJones::operator()(const idx_t& idx, data::EnergyAndVirialReducer& energyAndVirial) const
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

real_t LennardJones::getEnergy() const { return energyAndVirial_.energy; }
real_t LennardJones::getVirial() const { return energyAndVirial_.virial; }

LennardJones::LennardJones(const real_t rc,
                           const real_t& sigma,
                           const real_t& epsilon,
                           const real_t& cappingDistance)
    : LennardJones({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, false)
{
}

LennardJones::LennardJones(const std::vector<real_t>& cappingDistance,
                           const std::vector<real_t>& rc,
                           const std::vector<real_t>& sigma,
                           const std::vector<real_t>& epsilon,
                           const idx_t& numTypes,
                           const bool isShifted)
    : LJ_(cappingDistance, rc, sigma, epsilon, numTypes, isShifted), numTypes_(1)
{
    auto rcMax = std::ranges::max(rc);
    rcSqr_ = rcMax * rcMax;
}
}  // namespace mrmd::action

namespace mrmd::action::impl
{
KOKKOS_FUNCTION
CappedLennardJonesPotential::ForceAndEnergy CappedLennardJonesPotential::computeForceAndEnergy(
    const real_t& distSqr, const idx_t& typeIdx) const
{
    ForceAndEnergy ret;
    if (distSqr >= precomputedValues_(typeIdx).cappingDistanceSqr)
    {
        // normal LJ calculation
        auto frac2 = 1_r / distSqr;
        auto frac6 = frac2 * frac2 * frac2;
        ret.forceFactor =
            frac6 * (precomputedValues_(typeIdx).ff1 * frac6 - precomputedValues_(typeIdx).ff2) *
            frac2;
        ret.energy =
            frac6 * (precomputedValues_(typeIdx).ef1 * frac6 - precomputedValues_(typeIdx).ef2) -
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

KOKKOS_FUNCTION
void CappedLennardJonesPotential::operator()(const idx_t& typeIdx) const
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

CappedLennardJonesPotential::CappedLennardJonesPotential(const std::vector<real_t>& cappingDistance,
                                                         const std::vector<real_t>& rc,
                                                         const std::vector<real_t>& sigma,
                                                         const std::vector<real_t>& epsilon,
                                                         const idx_t& numTypes,
                                                         const bool isShifted)
    : isShifted_(isShifted)
{
    assert(idx_c(cappingDistance.size()) == numTypes * numTypes);
    assert(idx_c(rc.size()) == numTypes * numTypes);
    assert(idx_c(sigma.size()) == numTypes * numTypes);
    assert(idx_c(epsilon.size()) == numTypes * numTypes);

    precomputedValues_ = Kokkos::View<PrecomputedValues*>(
        "CappedLennardJonesPotential::PrecomputedValues", numTypes * numTypes);
    auto hPrecomputedValues = Kokkos::create_mirror_view(Kokkos::HostSpace(), precomputedValues_);

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
}  // namespace mrmd::action::impl