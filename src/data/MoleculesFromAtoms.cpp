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