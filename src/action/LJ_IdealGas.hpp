#pragma once

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
    const real_t epsilon_;
    real_t sig2_;
    real_t sig6_;
    real_t ff1_;
    real_t ff2_;
    real_t rcSqr_;

    data::Molecules::pos_t moleculesPos_;
    data::Molecules::lambda_t moleculesLambda_;
    data::Molecules::atoms_end_idx_t moleculesAtomEndIdx_;

    data::Particles::pos_t atomsPos_;
    data::Particles::force_t::atomic_access_slice atomsForce_;

    VerletList verletList_;

public:
    KOKKOS_INLINE_FUNCTION
    real_t computeCappedLJForce_(const real_t& distSqr) const
    {
        // normal LJ force calculation
        auto frac2 = 1.0 / distSqr;
        auto frac6 = frac2 * frac2 * frac2;
        auto ffactor = frac6 * (ff1_ * frac6 - ff2_) * frac2;

        // force capping
        ffactor = std::min(ffactor, +10_r);

        return ffactor;
    }

    /**
     * Loop over molecules
     *
     * @param alpha first molecule index
     */
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& alpha) const
    {
        /// epsilon for region checks
        constexpr auto eps = 0.001_r;

        /// weighting for molecule alpha
        auto lambdaAlpha = moleculesLambda_(alpha);
        assert(0_r <= lambdaAlpha);
        assert(lambdaAlpha <= 1_r);

        const auto numNeighbors = idx_c(NeighborList::numNeighbor(verletList_, alpha));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            /// second molecule index
            idx_t beta = idx_c(NeighborList::getNeighbor(verletList_, alpha, n));
            assert(0 <= beta);

            /// weighting for molecule beta
            auto lambdaBeta = moleculesLambda_(beta);
            assert(0_r <= lambdaBeta);
            assert(lambdaBeta <= 1_r);

            /// combined weighting of molecules alpha and beta
            auto weighting = 0.5_r * (lambdaAlpha + lambdaBeta);
            assert(0_r <= weighting);
            assert(weighting <= 1_r);
            if (weighting < eps)
            {
                // CG region -> ideal gas -> no interaction
                continue;
            }

            /// inclusive start index of atoms belonging to alpha
            auto startAtomsAlpha = alpha != 0 ? moleculesAtomEndIdx_(alpha - 1) : 0;
            /// exclusive end index of atoms belonging to alpha
            auto endAtomsAlpha = moleculesAtomEndIdx_(alpha);
            assert(0 <= startAtomsAlpha);
            assert(startAtomsAlpha < endAtomsAlpha);
            //            assert(endAtomsAlpha <= atoms_.numLocalParticles +
            //            atoms_.numGhostParticles);

            /// inclusive start index of atoms belonging to beta
            auto startAtomsBeta = beta != 0 ? moleculesAtomEndIdx_(beta - 1) : 0;
            /// exclusive end index of atoms belonging to beta
            auto endAtomsBeta = moleculesAtomEndIdx_(beta);
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

                real_t forceTmp[3] = {0_r, 0_r, 0_r};

                for (idx_t jdx = startAtomsBeta; jdx < endAtomsBeta; ++jdx)
                {
                    auto dx = posTmp[0] - atomsPos_(jdx, 0);
                    auto dy = posTmp[1] - atomsPos_(jdx, 1);
                    auto dz = posTmp[2] - atomsPos_(jdx, 2);

                    auto distSqr = dx * dx + dy * dy + dz * dz;

                    if (distSqr > rcSqr_) continue;

                    auto ffactor = computeCappedLJForce_(distSqr) * weighting;

                    forceTmp[0] += dx * ffactor;
                    forceTmp[1] += dy * ffactor;
                    forceTmp[2] += dz * ffactor;

                    atomsForce_(jdx, 0) -= dx * ffactor;
                    atomsForce_(jdx, 1) -= dy * ffactor;
                    atomsForce_(jdx, 2) -= dz * ffactor;
                }

                atomsForce_(idx, 0) += forceTmp[0];
                atomsForce_(idx, 1) += forceTmp[1];
                atomsForce_(idx, 2) += forceTmp[2];
            }
        }
    }

    LJ_IdealGas(const real_t& rc,
                const real_t& sigma,
                const real_t& epsilon,
                data::Molecules& molecules,
                VerletList& verletList,
                data::Particles& atoms)
        : epsilon_(epsilon), rcSqr_(rc * rc)
    {
        sig2_ = sigma * sigma;
        sig6_ = sig2_ * sig2_ * sig2_;
        ff1_ = 48.0 * epsilon * sig6_ * sig6_;
        ff2_ = 24.0 * epsilon * sig6_;

        moleculesPos_ = molecules.getPos();
        moleculesLambda_ = molecules.getLambda();
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
    static void applyForces(const real_t& rc,
                            const real_t& sigma,
                            const real_t& epsilon,
                            data::Molecules& molecules,
                            VerletList& verletList,
                            data::Particles& atoms)
    {
        impl::LJ_IdealGas forceModel(rc, sigma, epsilon, molecules, verletList, atoms);

        auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
        Kokkos::parallel_for(policy, forceModel, "LJ_IdealGas::applyForces");

        Kokkos::fence();
    }
};

}  // namespace action
}  // namespace mrmd