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

#include "ContributeMoleculeForceToAtoms.hpp"

namespace mrmd
{
namespace action
{
namespace ContributeMoleculeForceToAtoms
{
void update(const data::Molecules& molecules, const data::Atoms& atoms)
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

}  // namespace ContributeMoleculeForceToAtoms
}  // namespace action
}  // namespace mrmd