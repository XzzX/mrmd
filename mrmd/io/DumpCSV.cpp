#include "io/DumpCSV.hpp"

#include <fstream>
#include <iostream>

#include "data/Atoms.hpp"

namespace mrmd
{
namespace io
{
void dumpCSV(const std::string& filename, data::Atoms& atoms, bool dumpGhosts)
{
    data::HostAtoms at(atoms);
    auto pos = at.getPos();
    auto vel = at.getVel();
    auto type = at.getType();

    std::ofstream fout(filename);
    if (!fout.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    fout << "idx, mol, type, ghost, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z" << std::endl;
    auto lastAtomIdx = at.numLocalAtoms + (dumpGhosts ? at.numGhostAtoms : 0);
    for (idx_t idx = 0; idx < lastAtomIdx; ++idx)
    {
        fout << idx << ", " << idx / 3 << ", " << type(idx) << ", "
             << ((idx < at.numLocalAtoms) ? 0 : 1) << ", " << pos(idx, 0) << ", " << pos(idx, 1)
             << ", " << pos(idx, 2) << ", " << vel(idx, 0) << ", " << vel(idx, 1) << ", "
             << vel(idx, 2) << std::endl;
    }
    fout.close();
}
}  // namespace io
}  // namespace mrmd
