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

#include "MoleculesFromAtoms.hpp"

namespace mrmd::data
{
data::Molecules createMoleculeForEachAtom(data::Atoms& atoms)
{
    auto size = atoms.numLocalAtoms + atoms.numGhostAtoms;
    data::Molecules molecules(2 * size);
    auto atomsOffset = molecules.getAtomsOffset();
    auto numAtoms = molecules.getNumAtoms();

    auto policy = Kokkos::RangePolicy<>(0, size);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        atomsOffset(idx) = idx;
        numAtoms(idx) = 1;
    };
    Kokkos::parallel_for("createMoleculeForEachAtom", policy, kernel);
    Kokkos::fence();

    molecules.numLocalMolecules = atoms.numLocalAtoms;
    molecules.numGhostMolecules = atoms.numGhostAtoms;

    return molecules;
}
}  // namespace mrmd::data