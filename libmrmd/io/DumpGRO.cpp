#include "io/DumpGRO.hpp"

#include <fstream>

namespace mrmd
{
namespace io
{
/// format specification
/// https://manual.gromacs.org/current/reference-manual/file-formats.html?highlight=gro#gro
void dumpGRO(const std::string& filename,
             data::Atoms& atoms,
             const data::Subdomain& subdomain,
             const real_t& timestamp,
             const std::string& title,
             bool dumpGhosts,
             bool dumpVelocities)
{
    // very ugly, will also copy the whole atom data which is unnecessary, custom slicing
    // required
    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);

    std::ofstream fout(filename);
    if (!fout.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    auto lastAtomIdx = atoms.numLocalAtoms + (dumpGhosts ? atoms.numGhostAtoms : 0);
    fout << title << ", t=" << timestamp << std::endl;
    fout << lastAtomIdx << std::endl;
    for (idx_t idx = 0; idx < lastAtomIdx; ++idx)
    {
        char buf[1024];
        if (!dumpVelocities)
        {
            sprintf(buf,
                    "%5d%-5s%5s%5d%8.3f%8.3f%8.3f",
                    int_c(idx / 3 + 1),
                    "WATER",
                    (idx % 3) == 0 ? "OW1" : ((idx % 3) == 1 ? "HW2" : "HW3"),
                    int_c(idx + 1),
                    pos(idx, 0),
                    pos(idx, 1),
                    pos(idx, 2));
        }
        else
        {
            sprintf(buf,
                    "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f",
                    int_c(idx / 3 + 1),
                    "WATER",
                    (idx % 3) == 0 ? "OW1" : ((idx % 3) == 1 ? "HW2" : "HW3"),
                    int_c(idx + 1),
                    pos(idx, 0),
                    pos(idx, 1),
                    pos(idx, 2),
                    vel(idx, 0),
                    vel(idx, 1),
                    vel(idx, 2));
        }
        fout << std::string(buf) << std::endl;
    }
    fout << "    " << subdomain.diameter[0] << " " << subdomain.diameter[1] << " "
         << subdomain.diameter[2] << std::endl;
    fout.close();
}
}  // namespace io
}  // namespace mrmd