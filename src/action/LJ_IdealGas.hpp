#pragma once

#include <cassert>

#include "LennardJones.hpp"
#include "data/Histogram.hpp"
#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"
#include "weighting_function/CheckRegion.hpp"

namespace mrmd
{
namespace action
{
inline void updateMeanCompensationEnergy(data::Histogram& compensationEnergy,
                                         data::Histogram& compensationEnergyCounter,
                                         data::Histogram& meanCompensationEnergy,
                                         const real_t runningAverageFactor = 10_r)
{
    assert(compensationEnergy.numBins == compensationEnergyCounter.numBins);
    assert(compensationEnergy.numBins == meanCompensationEnergy.numBins);

    auto policy = Kokkos::RangePolicy<>(0, compensationEnergy.numBins);
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx)
    {
        // check if there is at least one entry in this bin
        if (compensationEnergyCounter.data(binIdx) < 0.5_r) return;
        assert(compensationEnergyCounter.data(binIdx) > 0);
        auto energy = compensationEnergy.data(binIdx) / compensationEnergyCounter.data(binIdx);

        // use running average to calculate new mean compensation energy
        meanCompensationEnergy.data(binIdx) =
            (runningAverageFactor * meanCompensationEnergy.data(binIdx) + energy) /
            (runningAverageFactor + 1_r);

        // reset accumulation histograms
        compensationEnergy.data(binIdx) = 0_r;
        compensationEnergyCounter.data(binIdx) = 0_r;
    };
    Kokkos::parallel_for("updateMeanCompensationEnergy", policy, kernel);
    Kokkos::fence();
}

class LJ_IdealGas
{
private:
    impl::CappedLennardJonesPotential LJ_;
    real_t rcSqr_ = 0_r;

    data::Molecules::pos_t moleculesPos_;
    data::Molecules::force_t::atomic_access_slice moleculesForce_;
    data::Molecules::lambda_t moleculesLambda_;
    data::Molecules::modulated_lambda_t moleculesModulatedLambda_;
    data::Molecules::grad_lambda_t moleculesGradLambda_;
    data::Molecules::atoms_end_idx_t moleculesAtomEndIdx_;

    data::Particles::pos_t atomsPos_;
    data::Particles::force_t::atomic_access_slice atomsForce_;

    data::Histogram compensationEnergy_ = data::Histogram("compensationEnergy", 0_r, 1_r, 200);
    ScalarScatterView compensationEnergyScatter_;

    data::Histogram compensationEnergyCounter_ =
        data::Histogram("compensationEnergyCounter", 0_r, 1_r, 200);

    data::Histogram meanCompensationEnergy_ =
        data::Histogram("meanCompensationEnergy", 0_r, 1_r, 200);

    bool isDriftCompensationSamplingRun_ = false;

    VerletList verletList_;

    idx_t runCounter_ = 0;

public:
    static constexpr idx_t COMPENSATION_ENERGY_SAMPLING_INTERVAL = 200;
    static constexpr idx_t COMPENSATION_ENERGY_UPDATE_INTERVAL = 20000;

    const auto& getMeanCompensationEnergy() const { return meanCompensationEnergy_; }

    /**
     * Loop over molecules
     *
     * @param alpha first molecule index
     */
    KOKKOS_INLINE_FUNCTION void operator()(const idx_t& alpha, real_t& sumEnergy) const
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

        const auto numNeighbors = idx_c(NeighborList::numNeighbor(verletList_, alpha));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            /// second molecule index
            const idx_t beta = idx_c(NeighborList::getNeighbor(verletList_, alpha, n));
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

            /// inclusive start index of atoms belonging to alpha
            const auto startAtomsAlpha = alpha != 0 ? moleculesAtomEndIdx_(alpha - 1) : 0;
            /// exclusive end index of atoms belonging to alpha
            const auto endAtomsAlpha = moleculesAtomEndIdx_(alpha);
            assert(0 <= startAtomsAlpha);
            assert(startAtomsAlpha < endAtomsAlpha);
            //            assert(endAtomsAlpha <= atoms_.numLocalParticles +
            //            atoms_.numGhostParticles);

            /// inclusive start index of atoms belonging to beta
            const auto startAtomsBeta = beta != 0 ? moleculesAtomEndIdx_(beta - 1) : 0;
            /// exclusive end index of atoms belonging to beta
            const auto endAtomsBeta = moleculesAtomEndIdx_(beta);
            assert(0 <= startAtomsBeta);
            assert(startAtomsBeta < endAtomsBeta);
            //            assert(endAtomsBeta <= atoms_.numLocalParticles +
            //            atoms_.numGhostParticles);

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

                    auto ffactor = LJ_.computeForce(distSqr) * weighting;

                    // if (ffactor > 10000_r)
                    //{
                    //     std::cout << "f: " << ffactor << " | " << std::sqrt(distSqr) <<
                    //     std::endl; std::cout << "x: " << posTmp[0] << " | " << atomsPos_(jdx, 0)
                    //     << " | "
                    //               << std::abs(dx) << std::endl;
                    //     std::cout << "y: " << posTmp[1] << " | " << atomsPos_(jdx, 1) << " | "
                    //               << std::abs(dy) << std::endl;
                    //     std::cout << "z: " << posTmp[2] << " | " << atomsPos_(jdx, 2) << " | "
                    //               << std::abs(dz) << std::endl;
                    // }

                    forceTmpIdx[0] += dx * ffactor;
                    forceTmpIdx[1] += dy * ffactor;
                    forceTmpIdx[2] += dz * ffactor;

                    atomsForce_(jdx, 0) -= dx * ffactor;
                    atomsForce_(jdx, 1) -= dy * ffactor;
                    atomsForce_(jdx, 2) -= dz * ffactor;

                    auto energy = LJ_.computeEnergy(distSqr);
                    sumEnergy += energy * weighting;
                    auto Vij = 0.5_r * energy;

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
                                    access(binAlpha) += Vij;
                                if (weighting_function::isInHYRegion(modulatedLambdaBeta) &&
                                    (binBeta != -1))
                                    access(binBeta) += Vij;
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
            if (weighting_function::isInHYRegion(modulatedLambdaAlpha) && (binAlpha != -1))
                compensationEnergyCounter_.data(binAlpha) += 1_r;
        }

        // drift force compensation
        if (weighting_function::isInHYRegion(modulatedLambdaAlpha) && (binAlpha != -1))
        {
            forceTmpAlpha[0] += meanCompensationEnergy_.data(binAlpha) * gradLambdaAlpha[0];
            forceTmpAlpha[1] += meanCompensationEnergy_.data(binAlpha) * gradLambdaAlpha[1];
            forceTmpAlpha[2] += meanCompensationEnergy_.data(binAlpha) * gradLambdaAlpha[2];
        }

        moleculesForce_(alpha, 0) += forceTmpAlpha[0];
        moleculesForce_(alpha, 1) += forceTmpAlpha[1];
        moleculesForce_(alpha, 2) += forceTmpAlpha[2];
    }

    real_t run(data::Molecules& molecules, VerletList& verletList, data::Particles& atoms)
    {
        moleculesPos_ = molecules.getPos();
        moleculesForce_ = molecules.getForce();
        moleculesLambda_ = molecules.getLambda();
        moleculesModulatedLambda_ = molecules.getModulatedLambda();
        moleculesGradLambda_ = molecules.getGradLambda();
        moleculesAtomEndIdx_ = molecules.getAtomsEndIdx();
        atomsPos_ = atoms.getPos();
        atomsForce_ = atoms.getForce();
        verletList_ = verletList;

        isDriftCompensationSamplingRun_ = runCounter_ % COMPENSATION_ENERGY_SAMPLING_INTERVAL == 0;

        compensationEnergyScatter_ = ScalarScatterView(compensationEnergy_.data);

        real_t energy = 0_r;
        auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
        Kokkos::parallel_reduce("LJ_IdealGas::applyForces", policy, *this, energy);

        Kokkos::Experimental::contribute(compensationEnergy_.data, compensationEnergyScatter_);

        Kokkos::fence();

        if (runCounter_ % COMPENSATION_ENERGY_UPDATE_INTERVAL == 0)
            updateMeanCompensationEnergy(
                compensationEnergy_, compensationEnergyCounter_, meanCompensationEnergy_, 10_r);

        ++runCounter_;

        return energy;
    }

    LJ_IdealGas(const real_t& cappingDistance,
                const real_t& rc,
                const real_t& sigma,
                const real_t& epsilon,
                const bool doShift)
        : LJ_(cappingDistance, rc, sigma, epsilon, doShift), rcSqr_(rc * rc)
    {
    }
};

}  // namespace action
}  // namespace mrmd