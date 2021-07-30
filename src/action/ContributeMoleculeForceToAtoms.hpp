#pragma once

#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class ContributeMoleculeForceToAtoms
{
public:
    static void update(const data::Molecules& molecules, const data::Particles& atoms)
    {
        auto moleculesForce = molecules.getForce();
        auto moleculesAtomEndIdx = molecules.getAtomsEndIdx();

        auto atomsForce = atoms.getForce();
        auto atomsRelativeMass = atoms.getRelativeMass();

        auto policy =
            Kokkos::RangePolicy<>(0, molecules.numLocalMolecules + molecules.numGhostMolecules);
        auto kernel = KOKKOS_LAMBDA(const idx_t& moleculeIdx)
        {
            auto atomsStart = moleculeIdx != 0 ? moleculesAtomEndIdx(moleculeIdx - 1) : 0;
            auto atomsEnd = moleculesAtomEndIdx(moleculeIdx);

            for (auto atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
            {
                atomsForce(atomIdx, 0) +=
                    atomsRelativeMass(atomIdx) * moleculesForce(moleculeIdx, 0);
                atomsForce(atomIdx, 1) +=
                    atomsRelativeMass(atomIdx) * moleculesForce(moleculeIdx, 1);
                atomsForce(atomIdx, 2) +=
                    atomsRelativeMass(atomIdx) * moleculesForce(moleculeIdx, 2);
            }
        };
        Kokkos::parallel_for(policy, kernel, "ContributeMoleculeForceToAtoms::update");
        Kokkos::fence();
    }
};
}  // namespace action
}  // namespace mrmd