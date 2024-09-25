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

#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "datatypes.hpp"

namespace mrmd::action::UpdateMolecules
{

template <typename WEIGHTING_FUNCTION>
static void update(const data::Molecules& molecules,
                   const data::Atoms& atoms,
                   const WEIGHTING_FUNCTION& weight)
{
    auto moleculesPos = molecules.getPos();
    auto moleculesLambda = molecules.getLambda();
    auto moleculesModulatedLambda = molecules.getModulatedLambda();
    auto moleculesGradLambda = molecules.getGradLambda();
    auto moleculesAtomsOffset = molecules.getAtomsOffset();
    auto moleculesNumAtoms = molecules.getNumAtoms();

    auto atomsPos = atoms.getPos();
    auto atomsRelativeMass = atoms.getRelativeMass();

    auto policy =
        Kokkos::RangePolicy<>(0, molecules.numLocalMolecules + molecules.numGhostMolecules);
    auto kernel = KOKKOS_LAMBDA(const idx_t& moleculeIdx)
    {
        auto atomsStart = moleculesAtomsOffset(moleculeIdx);
        auto atomsEnd = atomsStart + moleculesNumAtoms(moleculeIdx);

        moleculesPos(moleculeIdx, 0) = 0_r;
        moleculesPos(moleculeIdx, 1) = 0_r;
        moleculesPos(moleculeIdx, 2) = 0_r;

        for (auto atomIdx = atomsStart; atomIdx < atomsEnd; ++atomIdx)
        {
            assert(atomsRelativeMass(atomIdx) > 1e-8 &&
                   "Contribution almost zero. Did you forget to set relative mass?");
            moleculesPos(moleculeIdx, 0) += atomsPos(atomIdx, 0) * atomsRelativeMass(atomIdx);
            moleculesPos(moleculeIdx, 1) += atomsPos(atomIdx, 1) * atomsRelativeMass(atomIdx);
            moleculesPos(moleculeIdx, 2) += atomsPos(atomIdx, 2) * atomsRelativeMass(atomIdx);
        }

        weight(moleculesPos(moleculeIdx, 0),
               moleculesPos(moleculeIdx, 1),
               moleculesPos(moleculeIdx, 2),
               moleculesLambda(moleculeIdx),
               moleculesModulatedLambda(moleculeIdx),
               moleculesGradLambda(moleculeIdx, 0),
               moleculesGradLambda(moleculeIdx, 1),
               moleculesGradLambda(moleculeIdx, 2));
    };
    Kokkos::parallel_for("UpdateMolecules::update", policy, kernel);
    Kokkos::fence();
}

}  // namespace mrmd::action::UpdateMolecules
