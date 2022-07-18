#include "ContributeMoleculeForceToAtoms.hpp"

namespace mrmd
{
namespace action
{
void ContributeMoleculeForceToAtoms::update(const data::Molecules& molecules,
                                            const data::Atoms& atoms)
{
    auto moleculesForce = molecules.getForce();
    auto moleculesAtomsOffset = molecules.getAtomsOffset();
    auto moleculeNumAtoms = molecules.getNumAtoms();

    auto atomsForce = atoms.getForce();
    auto atomsRelativeMass = atoms.getRelativeMass();

    auto policy =
        Kokkos::RangePolicy<>(0, molecules.numLocalMolecules + molecules.numGhostMolecules);
    auto kernel = KOKKOS_LAMBDA(const idx_t& moleculeIdx)
    {
        auto atomsStart = moleculesAtomsOffset(moleculeIdx);
        auto atomsEnd = atomsStart + moleculeNumAtoms(moleculeIdx);

        for (auto atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
        {
            atomsForce(atomIdx, 0) += atomsRelativeMass(atomIdx) * moleculesForce(moleculeIdx, 0);
            atomsForce(atomIdx, 1) += atomsRelativeMass(atomIdx) * moleculesForce(moleculeIdx, 1);
            atomsForce(atomIdx, 2) += atomsRelativeMass(atomIdx) * moleculesForce(moleculeIdx, 2);
        }
    };
    Kokkos::parallel_for("ContributeMoleculeForceToAtoms::update", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd