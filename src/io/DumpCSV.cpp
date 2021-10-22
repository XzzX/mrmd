#include "io/DumpCSV.hpp"

#include <fstream>

#include "data/Atoms.hpp"

namespace mrmd
{
namespace io
{
void dumpCSV(const std::string& filename, data::Atoms& atoms, bool dumpGhosts)
{
    // very ugly, will also copy the whole atom data which is unnecessary, custom slicing
    // required
    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto type = Cabana::slice<data::Atoms::TYPE>(hAoSoA);

    std::ofstream fout(filename);
    if (!fout.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    fout << "idx, mol, type, ghost, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z" << std::endl;
    auto lastAtomIdx = atoms.numLocalAtoms + (dumpGhosts ? atoms.numGhostAtoms : 0);
    for (idx_t idx = 0; idx < lastAtomIdx; ++idx)
    {
        fout << idx << ", " << idx / 3 << ", " << type(idx) << ", "
             << ((idx < atoms.numLocalAtoms) ? 0 : 1) << ", " << pos(idx, 0) << ", " << pos(idx, 1)
             << ", " << pos(idx, 2) << ", " << vel(idx, 0) << ", " << vel(idx, 1) << ", "
             << vel(idx, 2) << std::endl;
    }
    fout.close();
}
}  // namespace io
}  // namespace mrmd