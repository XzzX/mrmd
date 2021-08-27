#pragma once

#include "CoulombDSF.hpp"
#include "LennardJones.hpp"
#include "Shake.hpp"
#include "data/Histogram.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"
#include "util/angle.hpp"

namespace mrmd
{
namespace action
{
class SPC
{
private:
    impl::CappedLennardJonesPotential LJ_;
    impl::CoulombDSF coulomb_;
    real_t rcSqr_ = 0_r;

    data::Molecules::pos_t moleculesPos_;
    data::Molecules::force_t::atomic_access_slice moleculesForce_;
    data::Molecules::lambda_t moleculesLambda_;
    data::Molecules::modulated_lambda_t moleculesModulatedLambda_;
    data::Molecules::grad_lambda_t moleculesGradLambda_;
    data::Molecules::atoms_offset_t moleculesAtomsOffset_;
    data::Molecules::num_atoms_t moleculesNumAtoms_;

    data::Particles::pos_t atomsPos_;
    data::Particles::force_t::atomic_access_slice atomsForce_;
    data::Particles::charge_t atomsCharge_;

    data::Histogram compensationEnergy_ = data::Histogram("compensationEnergy", 0_r, 1_r, 200);
    ScalarScatterView compensationEnergyScatter_;

    data::Histogram compensationEnergyCounter_ =
        data::Histogram("compensationEnergyCounter", 0_r, 1_r, 200);

    data::Histogram meanCompensationEnergy_ =
        data::Histogram("meanCompensationEnergy", 0_r, 1_r, 200);

    bool isDriftCompensationSamplingRun_ = false;

    VerletList verletList_;

    idx_t runCounter_ = 0;

    data::BondView::host_mirror_type bonds_;

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
        const auto numNeighbors = idx_c(NeighborList::numNeighbor(verletList_, alpha));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            /// second molecule index
            const idx_t beta = idx_c(NeighborList::getNeighbor(verletList_, alpha, n));
            assert(0 <= beta);

            /// inclusive start index of atoms belonging to alpha
            const auto startAtomsAlpha = moleculesAtomsOffset_(alpha);
            /// exclusive end index of atoms belonging to alpha
            const auto endAtomsAlpha = startAtomsAlpha + moleculesNumAtoms_(alpha);
            assert(0 <= startAtomsAlpha);
            assert(startAtomsAlpha < endAtomsAlpha);

            /// inclusive start index of atoms belonging to beta
            const auto startAtomsBeta = moleculesAtomsOffset_(beta);
            /// exclusive end index of atoms belonging to beta
            const auto endAtomsBeta = startAtomsBeta + moleculesNumAtoms_(beta);
            assert(0 <= startAtomsBeta);
            assert(startAtomsBeta < endAtomsBeta);

            // LJ interaction between oxygen atoms
            real_t posTmp[3];
            posTmp[0] = atomsPos_(startAtomsAlpha, 0);
            posTmp[1] = atomsPos_(startAtomsAlpha, 1);
            posTmp[2] = atomsPos_(startAtomsAlpha, 2);

            real_t dx[3];
            dx[0] = posTmp[0] - atomsPos_(startAtomsBeta, 0);
            dx[1] = posTmp[1] - atomsPos_(startAtomsBeta, 1);
            dx[2] = posTmp[2] - atomsPos_(startAtomsBeta, 2);

            const auto distSqr = util::dot3(dx, dx);

            if (distSqr > rcSqr_) continue;

            auto ffactor = LJ_.computeForce(distSqr);

            atomsForce_(startAtomsBeta, 0) -= dx[0] * ffactor;
            atomsForce_(startAtomsBeta, 1) -= dx[1] * ffactor;
            atomsForce_(startAtomsBeta, 2) -= dx[2] * ffactor;

            atomsForce_(startAtomsAlpha, 0) += dx[0] * ffactor;
            atomsForce_(startAtomsAlpha, 1) += dx[1] * ffactor;
            atomsForce_(startAtomsAlpha, 2) += dx[2] * ffactor;

            /// loop over atoms
            for (idx_t idx = startAtomsAlpha; idx < endAtomsAlpha; ++idx)
            {
                real_t posTmp[3];
                posTmp[0] = atomsPos_(idx, 0);
                posTmp[1] = atomsPos_(idx, 1);
                posTmp[2] = atomsPos_(idx, 2);

                auto q1 = atomsCharge_(idx);

                // avoid atomic force contributions to idx in innermost loop
                real_t forceTmpIdx[3] = {0_r, 0_r, 0_r};

                for (idx_t jdx = startAtomsBeta; jdx < endAtomsBeta; ++jdx)
                {
                    auto q2 = atomsCharge_(jdx);

                    real_t dx[3];
                    dx[0] = posTmp[0] - atomsPos_(jdx, 0);
                    dx[1] = posTmp[1] - atomsPos_(jdx, 1);
                    dx[2] = posTmp[2] - atomsPos_(jdx, 2);

                    const auto distSqr = util::dot3(dx, dx);

                    if (distSqr > rcSqr_) continue;

                    auto ffactor = coulomb_.computeForce(distSqr, q1, q2);

                    forceTmpIdx[0] += dx[0] * ffactor;
                    forceTmpIdx[1] += dx[1] * ffactor;
                    forceTmpIdx[2] += dx[2] * ffactor;

                    atomsForce_(jdx, 0) -= dx[0] * ffactor;
                    atomsForce_(jdx, 1) -= dx[1] * ffactor;
                    atomsForce_(jdx, 2) -= dx[2] * ffactor;
                }

                atomsForce_(idx, 0) += forceTmpIdx[0];
                atomsForce_(idx, 1) += forceTmpIdx[1];
                atomsForce_(idx, 2) += forceTmpIdx[2];
            }
        }
    }

    void enforceConstraints(data::Molecules& molecules, data::Particles& atoms, real_t dt)
    {
        MoleculeConstraints moleculeConstraints(3, 3);
        moleculeConstraints.setConstraints(bonds_);
        moleculeConstraints.enforcePositionalConstraints(molecules, atoms, dt);
    }

    real_t applyForces(data::Molecules& molecules, VerletList& verletList, data::Particles& atoms)
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
        atomsCharge_ = atoms.getCharge();
        verletList_ = verletList;

        isDriftCompensationSamplingRun_ = runCounter_ % COMPENSATION_ENERGY_SAMPLING_INTERVAL == 0;

        //        compensationEnergyScatter_ = ScalarScatterView(compensationEnergy_.data);

        real_t energy = 0_r;
        auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
        Kokkos::parallel_reduce("LJ_IdealGas::applyForces", policy, *this, energy);

        //        Kokkos::Experimental::contribute(compensationEnergy_.data,
        //        compensationEnergyScatter_);

        Kokkos::fence();

        //        if (runCounter_ % COMPENSATION_ENERGY_UPDATE_INTERVAL == 0)
        //            updateMeanCompensationEnergy(
        //                compensationEnergy_, compensationEnergyCounter_, meanCompensationEnergy_,
        //                10_r);
        //
        //        ++runCounter_;

        return energy;
    }

    SPC(const real_t& cappingDistance,
        const real_t& rc,
        const real_t& sigma,
        const real_t& epsilon,
        const bool doShift)
        : LJ_(cappingDistance, rc, sigma, epsilon, doShift),
          coulomb_(rc, 0.1_r),
          rcSqr_(rc * rc),
          bonds_("bonds", 3)
    {
        bonds_(0).idx = 0;
        bonds_(0).jdx = 1;
        bonds_(0).eqDistance = 1_r;
        bonds_(1).idx = 0;
        bonds_(1).jdx = 2;
        bonds_(1).eqDistance = 1_r;
        bonds_(2).idx = 1;
        bonds_(2).jdx = 2;
        // law of cosines
        bonds_(2).eqDistance = std::sqrt(2_r - 2_r * std::cos(util::degToRad(109.47_r)));
    }
};
}  // namespace action
}  // namespace mrmd