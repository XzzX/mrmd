#pragma once

#include "LennardJones.hpp"
#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
namespace impl
{
class LJ_IdealGas
{
private:
    CappedLennardJonesPotential LJ_;
    real_t rcSqr_ = 0_r;

    data::Molecules::pos_t moleculesPos_;
    data::Molecules::force_t::atomic_access_slice moleculesForce_;
    data::Molecules::lambda_t moleculesLambda_;
    data::Molecules::grad_lambda_t moleculesGradLambda_;
    data::Molecules::atoms_end_idx_t moleculesAtomEndIdx_;

    data::Particles::pos_t atomsPos_;
    data::Particles::force_t::atomic_access_slice atomsForce_;

    VerletList verletList_;

public:
    /**
     * Loop over molecules
     *
     * @param alpha first molecule index
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& alpha) const
    {
        /// epsilon for region checks
        constexpr auto eps = 0.00001_r;

        // avoid atomic force contributions to idx in innermost loop
        real_t forceTmpAlpha[3] = {0_r, 0_r, 0_r};

        /// weighting for molecule alpha
        const auto lambdaAlpha = moleculesLambda_(alpha);
        assert(0_r <= lambdaAlpha);
        assert(lambdaAlpha <= 1_r);

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
            const auto lambdaBeta = moleculesLambda_(beta);
            assert(0_r <= lambdaBeta);
            assert(lambdaBeta <= 1_r);

            const real_t gradLambdaBeta[3] = {moleculesGradLambda_(beta, 0),
                                              moleculesGradLambda_(beta, 1),
                                              moleculesGradLambda_(beta, 2)};

            /// combined weighting of molecules alpha and beta
            const auto weighting = 0.5_r * (lambdaAlpha + lambdaBeta);
            assert(0_r <= weighting);
            assert(weighting <= 1_r);
            if (weighting < eps)
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

                    forceTmpIdx[0] += dx * ffactor;
                    forceTmpIdx[1] += dy * ffactor;
                    forceTmpIdx[2] += dz * ffactor;

                    atomsForce_(jdx, 0) -= dx * ffactor;
                    atomsForce_(jdx, 1) -= dy * ffactor;
                    atomsForce_(jdx, 2) -= dz * ffactor;

                    // drift force contribution
                    auto Vij = 0.5_r * LJ_.computeEnergy(distSqr);

                    forceTmpAlpha[0] += -Vij;
                    forceTmpAlpha[1] += -Vij;
                    forceTmpAlpha[2] += -Vij;

                    forceTmpBeta[0] += -Vij;
                    forceTmpBeta[1] += -Vij;
                    forceTmpBeta[2] += -Vij;
                }

                atomsForce_(idx, 0) += forceTmpIdx[0];
                atomsForce_(idx, 1) += forceTmpIdx[1];
                atomsForce_(idx, 2) += forceTmpIdx[2];
            }

            moleculesForce_(beta, 0) += forceTmpBeta[0] * gradLambdaBeta[0];
            moleculesForce_(beta, 1) += forceTmpBeta[1] * gradLambdaBeta[1];
            moleculesForce_(beta, 2) += forceTmpBeta[2] * gradLambdaBeta[2];
        }
        moleculesForce_(alpha, 0) += forceTmpAlpha[0] * gradLambdaAlpha[0];
        moleculesForce_(alpha, 1) += forceTmpAlpha[1] * gradLambdaAlpha[1];
        moleculesForce_(alpha, 2) += forceTmpAlpha[2] * gradLambdaAlpha[2];
    }

    LJ_IdealGas(const real_t& cappingDistance,
                const real_t& rc,
                const real_t& sigma,
                const real_t& epsilon,
                data::Molecules& molecules,
                VerletList& verletList,
                data::Particles& atoms,
                const bool doShift)
        : LJ_(cappingDistance, rc, sigma, epsilon, doShift), rcSqr_(rc * rc)
    {
        moleculesPos_ = molecules.getPos();
        moleculesForce_ = molecules.getForce();
        moleculesLambda_ = molecules.getLambda();
        moleculesGradLambda_ = molecules.getGradLambda();
        moleculesAtomEndIdx_ = molecules.getAtomsEndIdx();
        atomsPos_ = atoms.getPos();
        atomsForce_ = atoms.getForce();
        verletList_ = verletList;
    }
};
}  // namespace impl

class LJ_IdealGas
{
public:
    static void applyForces(const real_t& cappingDistance,
                            const real_t& rc,
                            const real_t& sigma,
                            const real_t& epsilon,
                            data::Molecules& molecules,
                            VerletList& verletList,
                            data::Particles& atoms,
                            const bool doShift)
    {
        impl::LJ_IdealGas forceModel(
            cappingDistance, rc, sigma, epsilon, molecules, verletList, atoms, doShift);

        auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
        Kokkos::parallel_for(policy, forceModel, "LJ_IdealGas::applyForces");

        Kokkos::fence();
    }
};

}  // namespace action
}  // namespace mrmd