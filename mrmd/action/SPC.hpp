#pragma once

#include "Coulomb.hpp"
#include "LennardJones.hpp"
#include "Shake.hpp"
#include "data/Atoms.hpp"
#include "data/Histogram.hpp"
#include "datatypes.hpp"
#include "util/angle.hpp"

namespace mrmd::action::impl
{
struct Energy
{
    real_t LJ = real_t(0);
    real_t coulomb = real_t(0);

    KOKKOS_INLINE_FUNCTION
    Energy() = default;
    KOKKOS_INLINE_FUNCTION
    Energy(const Energy& rhs) = default;

    KOKKOS_INLINE_FUNCTION
    Energy& operator+=(const Energy& src)
    {
        LJ += src.LJ;
        coulomb += src.coulomb;
        return *this;
    }
    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile Energy& src) volatile
    {
        LJ += src.LJ;
        coulomb += src.coulomb;
    }
};
}  // namespace mrmd::action::impl

namespace Kokkos
{  // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<mrmd::action::impl::Energy>
{
    KOKKOS_FORCEINLINE_FUNCTION static mrmd::action::impl::Energy sum()
    {
        return mrmd::action::impl::Energy();
    }
};
}  // namespace Kokkos

namespace mrmd
{
namespace action
{
class SPC
{
private:
    impl::CappedLennardJonesPotential LJ_;
    impl::Coulomb coulomb_;
    real_t rcSqr_ = real_t(0);

    data::Molecules::pos_t moleculesPos_;
    data::Molecules::force_t::atomic_access_slice moleculesForce_;
    data::Molecules::lambda_t moleculesLambda_;
    data::Molecules::modulated_lambda_t moleculesModulatedLambda_;
    data::Molecules::grad_lambda_t moleculesGradLambda_;
    data::Molecules::atoms_offset_t moleculesAtomsOffset_;
    data::Molecules::num_atoms_t moleculesNumAtoms_;

    data::Atoms::pos_t atomsPos_;
    data::Atoms::force_t::atomic_access_slice atomsForce_;
    data::Atoms::charge_t atomsCharge_;

    data::Histogram compensationEnergy_ =
        data::Histogram("compensationEnergy", real_t(0), real_t(1), 200);
    ScalarScatterView compensationEnergyScatter_;

    data::Histogram compensationEnergyCounter_ =
        data::Histogram("compensationEnergyCounter", real_t(0), real_t(1), 200);

    data::Histogram meanCompensationEnergy_ =
        data::Histogram("meanCompensationEnergy", real_t(0), real_t(1), 200);

    bool isDriftCompensationSamplingRun_ = false;

    HalfVerletList verletList_;

    idx_t runCounter_ = 0;

    data::BondView::host_mirror_type bonds_;

public:
    real_t sumEnergyLJ_;
    real_t sumEnergyCoulomb_;

    auto getEnergyLJ() const { return sumEnergyLJ_; }
    auto getEnergyCoulomb() const { return sumEnergyCoulomb_; }

    static constexpr real_t massO = real_t(15.999);  ///< unit: g/mol
    static constexpr real_t chargeO = real_t(-0.82);

    static constexpr real_t massH = real_t(1.008);  ///< unit: g/mol
    static constexpr real_t chargeH = real_t(+0.41);

    // LJ parameters for O-O interaction
    // DOI: 10.1021/j100308a038
    static constexpr real_t sigma = real_t(0.31655578901998815);   ///< unit: nm
    static constexpr real_t epsilon = real_t(0.6501695808187486);  ///< unit: kJ / mol
    static constexpr real_t rc = real_t(1.2);                      ///< unit: nm

    // Coulomb DSF parameters
    static constexpr real_t alpha = real_t(2.0);  ///< unit: 1/nm

    /// unit: nm, equilibirum distance between hydrogen and oxygen
    static constexpr real_t eqDistanceHO = real_t(0.1);
    static constexpr real_t angleHOH = util::degToRad(real_t(109.47));  ///< unit: radians
    const real_t eqDistanceHH =
        eqDistanceHO * std::sqrt(real_t(2) - real_t(2) * std::cos(angleHOH));

    static constexpr idx_t COMPENSATION_ENERGY_SAMPLING_INTERVAL = 200;
    static constexpr idx_t COMPENSATION_ENERGY_UPDATE_INTERVAL = 20000;

    const auto& getMeanCompensationEnergy() const { return meanCompensationEnergy_; }

    struct BondEnergy
    {
    };
    struct CalcInteractions
    {
    };

    /**
     * Loop over molecules
     *
     * @param alpha first molecule index
     */
    KOKKOS_INLINE_FUNCTION void operator()(CalcInteractions,
                                           const idx_t& alpha,
                                           impl::Energy& sumEnergy) const
    {
        const auto numNeighbors = idx_c(HalfNeighborList::numNeighbor(verletList_, alpha));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            /// second molecule index
            const idx_t beta = idx_c(HalfNeighborList::getNeighbor(verletList_, alpha, n));
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

            if (distSqr < rcSqr_)
            {
                auto forceAndVirial = LJ_.computeForceAndEnergy(distSqr, 0);
                sumEnergy.LJ += forceAndVirial.energy;

                atomsForce_(startAtomsBeta, 0) -= dx[0] * forceAndVirial.forceFactor;
                atomsForce_(startAtomsBeta, 1) -= dx[1] * forceAndVirial.forceFactor;
                atomsForce_(startAtomsBeta, 2) -= dx[2] * forceAndVirial.forceFactor;

                atomsForce_(startAtomsAlpha, 0) += dx[0] * forceAndVirial.forceFactor;
                atomsForce_(startAtomsAlpha, 1) += dx[1] * forceAndVirial.forceFactor;
                atomsForce_(startAtomsAlpha, 2) += dx[2] * forceAndVirial.forceFactor;
            }

            /// loop over atoms
            for (idx_t idx = startAtomsAlpha; idx < endAtomsAlpha; ++idx)
            {
                real_t posTmp[3];
                posTmp[0] = atomsPos_(idx, 0);
                posTmp[1] = atomsPos_(idx, 1);
                posTmp[2] = atomsPos_(idx, 2);

                auto q1 = atomsCharge_(idx);

                // avoid atomic force contributions to idx in innermost loop
                real_t forceTmpIdx[3] = {real_t(0), real_t(0), real_t(0)};

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
                    sumEnergy.coulomb += coulomb_.computeEnergy(distSqr, q1, q2);

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

    void enforcePositionalConstraints(data::Molecules& molecules, data::Atoms& atoms, real_t dt)
    {
        MoleculeConstraints moleculeConstraints(3, 20);
        moleculeConstraints.setConstraints(bonds_);
        moleculeConstraints.enforcePositionalConstraints(molecules, atoms, dt);
    }

    void enforceVelocityConstraints(data::Molecules& molecules, data::Atoms& atoms, real_t dt)
    {
        MoleculeConstraints moleculeConstraints(3, 20);
        moleculeConstraints.setConstraints(bonds_);
        moleculeConstraints.enforceVelocityConstraints(molecules, atoms, dt);
    }

    void applyForces(data::Molecules& molecules, HalfVerletList& verletList, data::Atoms& atoms)
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

        auto energy = impl::Energy();
        auto policy = Kokkos::RangePolicy<CalcInteractions>(0, molecules.numLocalMolecules);
        Kokkos::parallel_reduce(
            "SPC::applyForces", policy, *this, Kokkos::Sum<impl::Energy>(energy));

        //        Kokkos::Experimental::contribute(compensationEnergy_.data,
        //        compensationEnergyScatter_);

        Kokkos::fence();

        //        if (runCounter_ % compensationEnergyUpdateInveral == 0)
        //            updateMeanCompensationEnergy(
        //                compensationEnergy_, compensationEnergyCounter_, meanCompensationEnergy_,
        //                1real_t(0));
        //
        //        ++runCounter_;

        sumEnergyLJ_ = energy.LJ;
        sumEnergyCoulomb_ = energy.coulomb;
    }

    KOKKOS_INLINE_FUNCTION void operator()(BondEnergy, const idx_t& alpha, real_t& sumEnergy) const
    {
        /// inclusive start index of atoms belonging to alpha
        const auto startAtomsAlpha = moleculesAtomsOffset_(alpha);
        assert(0 <= startAtomsAlpha);

        auto idxO = startAtomsAlpha;
        auto idxH0 = startAtomsAlpha + 1;
        auto idxH1 = startAtomsAlpha + 2;

        real_t dx[3];
        real_t dist = real_t(0);

        dx[0] = atomsPos_(idxO, 0) - atomsPos_(idxH0, 0);
        dx[1] = atomsPos_(idxO, 1) - atomsPos_(idxH0, 1);
        dx[2] = atomsPos_(idxO, 2) - atomsPos_(idxH0, 2);
        dist = std::sqrt(util::dot3(dx, dx));
        sumEnergy += util::sqr(dist - eqDistanceHO);

        dx[0] = atomsPos_(idxO, 0) - atomsPos_(idxH1, 0);
        dx[1] = atomsPos_(idxO, 1) - atomsPos_(idxH1, 1);
        dx[2] = atomsPos_(idxO, 2) - atomsPos_(idxH1, 2);
        dist = std::sqrt(util::dot3(dx, dx));
        sumEnergy += util::sqr(dist - eqDistanceHO);

        dx[0] = atomsPos_(idxH0, 0) - atomsPos_(idxH1, 0);
        dx[1] = atomsPos_(idxH0, 1) - atomsPos_(idxH1, 1);
        dx[2] = atomsPos_(idxH0, 2) - atomsPos_(idxH1, 2);
        dist = std::sqrt(util::dot3(dx, dx));
        sumEnergy += util::sqr(dist - eqDistanceHH);
    }

    real_t calcBondEnergy(data::Molecules& molecules,
                          data::Atoms& atoms,
                          const real_t& harmonicPreFactor)
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

        real_t energy = real_t(0);
        auto policy = Kokkos::RangePolicy<BondEnergy>(
            0, molecules.numLocalMolecules + molecules.numGhostMolecules);
        Kokkos::parallel_reduce("SPC::calcHarmonicPotential", policy, *this, energy);

        Kokkos::fence();

        return harmonicPreFactor * energy / real_c(atoms.numLocalAtoms + atoms.numGhostAtoms);
    }

    SPC()
        : LJ_({real_t(0.7) * sigma}, {rc}, {sigma}, {epsilon}, 1, true),
          coulomb_(),
          rcSqr_(rc * rc),
          bonds_("bonds", 3)
    {
        bonds_(0).idx = 0;
        bonds_(0).jdx = 1;
        bonds_(0).eqDistance = eqDistanceHO;
        bonds_(1).idx = 0;
        bonds_(1).jdx = 2;
        bonds_(1).eqDistance = eqDistanceHO;
        bonds_(2).idx = 1;
        bonds_(2).jdx = 2;
        bonds_(2).eqDistance = eqDistanceHH;
    }
};
}  // namespace action
}  // namespace mrmd