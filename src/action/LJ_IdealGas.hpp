#pragma once

#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
namespace impl
{
template <typename WEIGHTING_FUNCTION>
class LJ_IdealGas
{
private:
    const real_t epsilon_;
    real_t sig2_;
    real_t sig6_;
    real_t ff1_;
    real_t ff2_;
    real_t rcSqr_;

    data::Particles::pos_t moleculesPos_;
    data::Particles::offset_t atomOffsets_;

    data::Particles::pos_t atomsPos_;
    data::Particles::force_t::atomic_access_slice atomsForce_;

    VerletList verletList_;

    WEIGHTING_FUNCTION weightingFunction_;

public:
    KOKKOS_INLINE_FUNCTION
    real_t computeForce_(const real_t& distSqr) const
    {
        auto frac2 = 1.0 / distSqr;
        auto frac6 = frac2 * frac2 * frac2;
        return frac6 * (ff1_ * frac6 - ff2_) * frac2;
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
        auto lambdaAlpha = weightingFunction_(
            moleculesPos_(alpha, 0), moleculesPos_(alpha, 1), moleculesPos_(alpha, 2));
        assert(0_r <= lambdaAlpha);
        assert(lambdaAlpha <= 1_r);

        const auto numNeighbors = idx_c(NeighborList::numNeighbor(verletList_, alpha));
        for (idx_t n = 0; n < numNeighbors; ++n)
        {
            /// second molecule index
            idx_t beta = idx_c(NeighborList::getNeighbor(verletList_, alpha, n));
            assert(0 <= beta);

            /// weighting for molecule beta
            auto lambdaBeta = weightingFunction_(
                moleculesPos_(beta, 0), moleculesPos_(beta, 1), moleculesPos_(beta, 2));
            assert(0_r <= lambdaBeta);
            assert(lambdaBeta <= 1_r);

            /// combined weighting of molecules alpha and beta
            auto weighting = lambdaAlpha * lambdaBeta;
            assert(0_r <= weighting);
            assert(weighting <= 1_r);
            if (weighting < eps)
            {
                // CG region -> ideal gas -> no interaction
                continue;
            }

            /// inclusive start index of atoms belonging to alpha
            auto startAtomsAlpha = alpha != 0 ? atomOffsets_(alpha - 1) : 0;
            /// exclusive end index of atoms belonging to alpha
            auto endAtomsAlpha = atomOffsets_(alpha);
            assert(0 <= startAtomsAlpha);
            assert(startAtomsAlpha < endAtomsAlpha);
            //            assert(endAtomsAlpha <= atoms_.numLocalParticles +
            //            atoms_.numGhostParticles);

            /// inclusive start index of atoms belonging to beta
            auto startAtomsBeta = beta != 0 ? atomOffsets_(beta - 1) : 0;
            /// exclusive end index of atoms belonging to beta
            auto endAtomsBeta = atomOffsets_(beta);
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

                    auto ffactor = computeForce_(distSqr) * weighting;

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
                data::Particles& molecules,
                VerletList& verletList,
                data::Particles& atoms,
                WEIGHTING_FUNCTION& func)
        : epsilon_(epsilon), rcSqr_(rc * rc), weightingFunction_(func)
    {
        sig2_ = sigma * sigma;
        sig6_ = sig2_ * sig2_ * sig2_;
        ff1_ = 48.0 * epsilon * sig6_ * sig6_;
        ff2_ = 24.0 * epsilon * sig6_;

        moleculesPos_ = molecules.getPos();
        atomOffsets_ = molecules.getOffset();
        atomsPos_ = atoms.getPos();
        atomsForce_ = atoms.getForce();
        verletList_ = verletList;
    }
};
}  // namespace impl

class LJ_IdealGas
{
public:
    template <typename WEIGHTING_FUNCTION>
    static void applyForces(const real_t rc,
                            const real_t& sigma,
                            const real_t& epsilon,
                            data::Particles& molecules,
                            VerletList& verletList,
                            data::Particles& atoms,
                            WEIGHTING_FUNCTION& func)
    {
        impl::LJ_IdealGas forceModel(rc, sigma, epsilon, molecules, verletList, atoms, func);

        auto policy = Kokkos::RangePolicy<>(0, molecules.numLocalParticles);
        Kokkos::parallel_for(policy, forceModel, "LJ_IdealGas::applyForces");

        Kokkos::fence();
    }
};

}  // namespace action
}  // namespace mrmd