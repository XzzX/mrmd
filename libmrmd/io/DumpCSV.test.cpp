#include "DumpCSV.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"

namespace mrmd
{
namespace io
{
TEST(DumpCSV, atoms)
{
    auto atoms = data::Atoms(100 * 2);
    atoms.numLocalAtoms = 10;
    atoms.numGhostAtoms = 10;
    dumpCSV("test.csv", atoms, true);
}
}  // namespace io
}  // namespace mrmd