#pragma once

#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class UpdateMolecules
{
public:
    template <typename WEIGHTING_FUNCTION>
    static void update(const data::Molecules& molecules,
                       const data::Particles& atoms,
                       const WEIGHTING_FUNCTION& weight)
    {
        auto moleculesPos = molecules.getPos();
        auto moleculesLambda = molecules.getLambda();
        auto moleculesGradLambda = molecules.getGradLambda();
        auto moleculesAtomEndIdx = molecules.getAtomsEndIdx();

        auto atomsPos = atoms.getPos();
        auto atomsRelativeMass = atoms.getRelativeMass();

        auto policy =
            Kokkos::RangePolicy<>(0, molecules.numLocalMolecules + molecules.numGhostMolecules);
        auto kernel = KOKKOS_LAMBDA(const idx_t& moleculeIdx)
        {
            auto atomsStart = moleculeIdx != 0 ? moleculesAtomEndIdx(moleculeIdx - 1) : 0;
            auto atomsEnd = moleculesAtomEndIdx(moleculeIdx);

            moleculesPos(moleculeIdx, 0) = 0_r;
            moleculesPos(moleculeIdx, 1) = 0_r;
            moleculesPos(moleculeIdx, 2) = 0_r;

            for (auto atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
            {
                moleculesPos(moleculeIdx, 0) += atomsPos(atomIdx, 0) * atomsRelativeMass(atomIdx);
                moleculesPos(moleculeIdx, 1) += atomsPos(atomIdx, 1) * atomsRelativeMass(atomIdx);
                moleculesPos(moleculeIdx, 2) += atomsPos(atomIdx, 2) * atomsRelativeMass(atomIdx);
            }

            weight(moleculesPos(moleculeIdx, 0),
                   moleculesPos(moleculeIdx, 1),
                   moleculesPos(moleculeIdx, 2),
                   moleculesLambda(moleculeIdx),
                   moleculesGradLambda(moleculeIdx, 0),
                   moleculesGradLambda(moleculeIdx, 0),
                   moleculesGradLambda(moleculeIdx, 0));
        };
        Kokkos::parallel_for(policy, kernel, "UpdateMolecules::update");
        Kokkos::fence();
    }
};
}  // namespace action
}  // namespace mrmd