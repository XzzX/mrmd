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

#include "LJ_IdealGas.hpp"

#include <algorithm>

namespace mrmd::action
{
void updateMeanCompensationEnergy(data::MultiHistogram& compensationEnergy,
                                  data::MultiHistogram& compensationEnergyCounter,
                                  data::MultiHistogram& meanCompensationEnergy,
                                  const real_t runningAverageFactor)
{
    assert(compensationEnergy.numBins == compensationEnergyCounter.numBins);
    assert(compensationEnergy.numBins == meanCompensationEnergy.numBins);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {compensationEnergy.numBins, compensationEnergy.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        // check if there is at least one entry in this bin
        if (compensationEnergyCounter.data(binIdx, histogramIdx) < 0.5_r) return;
        assert(compensationEnergyCounter.data(binIdx, histogramIdx) > 0);
        auto energy = compensationEnergy.data(binIdx, histogramIdx) /
                      compensationEnergyCounter.data(binIdx, histogramIdx);

        // use running average to calculate new mean compensation energy
        meanCompensationEnergy.data(binIdx, histogramIdx) =
            (runningAverageFactor * meanCompensationEnergy.data(binIdx, histogramIdx) + energy) /
            (runningAverageFactor + 1_r);

        // reset accumulation histograms
        compensationEnergy.data(binIdx, histogramIdx) = 0_r;
        compensationEnergyCounter.data(binIdx, histogramIdx) = 0_r;
    };
    Kokkos::parallel_for("updateMeanCompensationEnergy", policy, kernel);
    Kokkos::fence();
}

KOKKOS_FUNCTION void LJ_IdealGas::operator()(const idx_t& alpha, real_t& sumEnergy) const
{
    // avoid atomic force contributions to idx in innermost loop
    real_t forceTmpAlpha[3] = {0_r, 0_r, 0_r};

    /// weighting for molecule alpha
    const auto modulatedLambdaAlpha = moleculesModulatedLambda_(alpha);
    assert(0_r <= modulatedLambdaAlpha);
    assert(modulatedLambdaAlpha <= 1_r);

    idx_t binAlpha = -1;
    if (weighting_function::isInHYRegion(modulatedLambdaAlpha))
        binAlpha = compensationEnergy_.getBin(moleculesLambda_(alpha));

    const real_t gradLambdaAlpha[3] = {moleculesGradLambda_(alpha, 0),
                                       moleculesGradLambda_(alpha, 1),
                                       moleculesGradLambda_(alpha, 2)};

    /// inclusive start index of atoms belonging to alpha
    const auto startAtomsAlpha = moleculesAtomsOffset_(alpha);
    /// exclusive end index of atoms belonging to alpha
    const auto endAtomsAlpha = startAtomsAlpha + moleculesNumAtoms_(alpha);
    assert(0 <= startAtomsAlpha);
    assert(startAtomsAlpha < endAtomsAlpha);
    //            assert(endAtomsAlpha <= atoms_.numLocalAtoms +
    //            atoms_.numGhostAtoms);

    const auto numNeighbors = idx_c(HalfNeighborList::numNeighbor(verletList_, alpha));
    for (idx_t n = 0; n < numNeighbors; ++n)
    {
        /// second molecule index
        const idx_t beta = idx_c(HalfNeighborList::getNeighbor(verletList_, alpha, n));
        assert(0 <= beta);

        // avoid atomic force contributions to idx in innermost loop
        real_t forceTmpBeta[3] = {0_r, 0_r, 0_r};

        /// weighting for molecule beta
        const auto modulatedLambdaBeta = moleculesModulatedLambda_(beta);
        assert(0_r <= modulatedLambdaBeta);
        assert(modulatedLambdaBeta <= 1_r);

        const real_t gradLambdaBeta[3] = {moleculesGradLambda_(beta, 0),
                                          moleculesGradLambda_(beta, 1),
                                          moleculesGradLambda_(beta, 2)};

        /// combined weighting of molecules alpha and beta
        const auto weighting = 0.5_r * (modulatedLambdaAlpha + modulatedLambdaBeta);
        assert(0_r <= weighting);
        assert(weighting <= 1_r);
        if (weighting_function::isInCGRegion(modulatedLambdaAlpha) &&
            weighting_function::isInCGRegion(modulatedLambdaBeta))
        {
            // CG region -> ideal gas -> no interaction
            continue;
        }

        /// inclusive start index of atoms belonging to beta
        const auto startAtomsBeta = moleculesAtomsOffset_(beta);
        /// exclusive end index of atoms belonging to beta
        const auto endAtomsBeta = startAtomsBeta + moleculesNumAtoms_(beta);
        assert(0 <= startAtomsBeta);
        assert(startAtomsBeta < endAtomsBeta);
        //            assert(endAtomsBeta <= atoms_.numLocalAtoms +
        //            atoms_.numGhostAtoms);

        /// loop over atoms
        for (idx_t idx = startAtomsAlpha; idx < endAtomsAlpha; ++idx)
        {
            real_t posTmp[3];
            posTmp[0] = atomsPos_(idx, 0);
            posTmp[1] = atomsPos_(idx, 1);
            posTmp[2] = atomsPos_(idx, 2);

            // avoid atomic force contributions to idx in innermost loop
            real_t forceTmpIdx[3] = {0_r, 0_r, 0_r};

            for (idx_t jdx = startAtomsBeta; jdx < endAtomsBeta; ++jdx)
            {
                const auto dx = posTmp[0] - atomsPos_(jdx, 0);
                const auto dy = posTmp[1] - atomsPos_(jdx, 1);
                const auto dz = posTmp[2] - atomsPos_(jdx, 2);

                const auto distSqr = dx * dx + dy * dy + dz * dz;

                if (distSqr > rcSqr_) continue;

                auto typeIdx = atomsType_(idx) * numTypes_ + atomsType_(jdx);
                MRMD_DEVICE_ASSERT_GREATEREQUAL(typeIdx, 0);
                MRMD_DEVICE_ASSERT_LESS(typeIdx, numTypes_ * numTypes_);
                MRMD_DEVICE_ASSERT(!std::isnan(distSqr));
                auto forceAndEnergy = LJ_.computeForceAndEnergy(distSqr, typeIdx);
                auto ffactor = forceAndEnergy.forceFactor * weighting;
                MRMD_DEVICE_ASSERT(!std::isnan(ffactor));

                forceTmpIdx[0] += dx * ffactor;
                forceTmpIdx[1] += dy * ffactor;
                forceTmpIdx[2] += dz * ffactor;

                atomsForce_(jdx, 0) -= dx * ffactor;
                atomsForce_(jdx, 1) -= dy * ffactor;
                atomsForce_(jdx, 2) -= dz * ffactor;

                MRMD_DEVICE_ASSERT(!std::isnan(forceAndEnergy.energy));
                sumEnergy += forceAndEnergy.energy * weighting;
                auto Vij = 0.5_r * forceAndEnergy.energy;

                if (weighting_function::isInHYRegion(modulatedLambdaAlpha) ||
                    weighting_function::isInHYRegion(modulatedLambdaBeta))
                {
                    // drift force contribution
                    forceTmpAlpha[0] += -Vij * gradLambdaAlpha[0];
                    forceTmpAlpha[1] += -Vij * gradLambdaAlpha[1];
                    forceTmpAlpha[2] += -Vij * gradLambdaAlpha[2];

                    forceTmpBeta[0] += -Vij * gradLambdaBeta[0];
                    forceTmpBeta[1] += -Vij * gradLambdaBeta[1];
                    forceTmpBeta[2] += -Vij * gradLambdaBeta[2];

                    // building histogram for drift force compensation
                    if (isDriftCompensationSamplingRun_)
                    {
                        idx_t binBeta = -1;
                        if (weighting_function::isInHYRegion(modulatedLambdaBeta))
                        {
                            binBeta = compensationEnergy_.getBin(moleculesLambda_(beta));
                        }
                        {
                            auto access = compensationEnergyScatter_.access();
                            if (weighting_function::isInHYRegion(modulatedLambdaAlpha) &&
                                (binAlpha != -1))
                                access(binAlpha, atomsType_(idx)) += Vij;
                            if (weighting_function::isInHYRegion(modulatedLambdaBeta) &&
                                (binBeta != -1))
                                access(binBeta, atomsType_(jdx)) += Vij;
                        }
                    }
                }
            }

            atomsForce_(idx, 0) += forceTmpIdx[0];
            atomsForce_(idx, 1) += forceTmpIdx[1];
            atomsForce_(idx, 2) += forceTmpIdx[2];
        }

        moleculesForce_(beta, 0) += forceTmpBeta[0];
        moleculesForce_(beta, 1) += forceTmpBeta[1];
        moleculesForce_(beta, 2) += forceTmpBeta[2];
    }

    if (isDriftCompensationSamplingRun_)
    {
        for (idx_t atomIdx = startAtomsAlpha; atomIdx < endAtomsAlpha; ++atomIdx)
        {
            if (weighting_function::isInHYRegion(modulatedLambdaAlpha) && (binAlpha != -1))
                compensationEnergyCounter_.data(binAlpha, atomsType_(atomIdx)) += 1_r;
        }
    }

    // drift force compensation
    if (weighting_function::isInHYRegion(modulatedLambdaAlpha) && (binAlpha != -1))
    {
        forceTmpAlpha[0] += meanCompensationEnergy_.data(binAlpha, atomsType_(startAtomsAlpha)) *
                            gradLambdaAlpha[0];
        forceTmpAlpha[1] += meanCompensationEnergy_.data(binAlpha, atomsType_(startAtomsAlpha)) *
                            gradLambdaAlpha[1];
        forceTmpAlpha[2] += meanCompensationEnergy_.data(binAlpha, atomsType_(startAtomsAlpha)) *
                            gradLambdaAlpha[2];
    }

    moleculesForce_(alpha, 0) += forceTmpAlpha[0];
    moleculesForce_(alpha, 1) += forceTmpAlpha[1];
    moleculesForce_(alpha, 2) += forceTmpAlpha[2];
}

real_t LJ_IdealGas::run(data::Molecules& molecules, HalfVerletList& verletList, data::Atoms& atoms)
{
    moleculesPos_ = molecules.getPos();
    moleculesForce_ = molecules.getForce();
    moleculesLambda_ = molecules.getLambda();
    moleculesModulatedLambda_ = molecules.getModulatedLambda();
    moleculesGradLambda_ = molecules.getGradLambda();
    moleculesAtomsOffset_ = molecules.getAtomsOffset();
    moleculesNumAtoms_ = molecules.getNumAtoms();
    atomsPos_ = atoms.getPos();
    atomsForce_ = atoms.getForce();
    atomsType_ = atoms.getType();
    verletList_ = verletList;

    isDriftCompensationSamplingRun_ = runCounter_ % compensationEnergySamplingInterval == 0;

    compensationEnergyScatter_ = MultiScatterView(compensationEnergy_.data);

    real_t energy = 0_r;
    auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
    Kokkos::parallel_reduce("LJ_IdealGas::applyForces", policy, *this, energy);

    Kokkos::Experimental::contribute(compensationEnergy_.data, compensationEnergyScatter_);

    Kokkos::fence();

    if (runCounter_ % compensationEnergyUpdateInveral == 0)
        updateMeanCompensationEnergy(
            compensationEnergy_, compensationEnergyCounter_, meanCompensationEnergy_, 10_r);

    ++runCounter_;

    return energy;
}

LJ_IdealGas::LJ_IdealGas(const real_t& cappingDistance,
                         const real_t& rc,
                         const real_t& sigma,
                         const real_t& epsilon,
                         const bool doShift)
    : LJ_IdealGas({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, doShift)
{
}

LJ_IdealGas::LJ_IdealGas(const std::vector<real_t>& cappingDistance,
                         const std::vector<real_t>& rc,
                         const std::vector<real_t>& sigma,
                         const std::vector<real_t>& epsilon,
                         const idx_t numTypes,
                         const bool doShift)
    : numTypes_(numTypes),
      LJ_(cappingDistance, rc, sigma, epsilon, numTypes, doShift),
      compensationEnergy_("compensationEnergy", 0_r, 1_r, COMPENSATION_ENERGY_BINS, numTypes),
      compensationEnergyCounter_(
          "compensationEnergyCounter", 0_r, 1_r, COMPENSATION_ENERGY_BINS, numTypes),
      meanCompensationEnergy_(
          "meanCompensationEnergy", 0_r, 1_r, COMPENSATION_ENERGY_BINS, numTypes)
{
    MRMD_HOST_ASSERT_EQUAL(cappingDistance.size(), numTypes * numTypes);
    MRMD_HOST_ASSERT_EQUAL(rc.size(), numTypes * numTypes);
    MRMD_HOST_ASSERT_EQUAL(sigma.size(), numTypes * numTypes);
    MRMD_HOST_ASSERT_EQUAL(epsilon.size(), numTypes * numTypes);

    auto maxRC = std::ranges::max(rc);
    rcSqr_ = maxRC * maxRC;
}

}  // namespace mrmd::action