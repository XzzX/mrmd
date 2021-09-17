#include "io/DumpGRO.hpp"

#include <fstream>

namespace mrmd
{
namespace io
{
/// format specification https://manual.gromacs.org/archive/5.0.4/online/gro.html
void dumpGRO(const std::string& filename,
             data::Particles& particles,
             const data::Subdomain& subdomain,
             const real_t& timestamp,
             const std::string& title,
             bool dumpGhosts)
{
    // very ugly, will also copy the whole particle data which is unnecessary, custom slicing
    // required
    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), particles.getAoSoA());
    auto pos = Cabana::slice<data::Particles::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Particles::VEL>(hAoSoA);

    std::ofstream fout(filename);
    if (!fout.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    auto lastParticleIdx =
        particles.numLocalParticles + (dumpGhosts ? particles.numGhostParticles : 0);
    fout << title << ", t=" << timestamp << std::endl;
    fout << lastParticleIdx << std::endl;
    for (idx_t idx = 0; idx < lastParticleIdx; ++idx)
    {
        char buf[1024];
        sprintf(buf,
                "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f",
                int_c(idx),
                "H",
                "H",
                int_c(idx),
                pos(idx, 0),
                pos(idx, 1),
                pos(idx, 2),
                vel(idx, 0),
                vel(idx, 1),
                vel(idx, 2));
        fout << std::string(buf) << std::endl;
    }
    fout << subdomain.diameter[0] << " " << subdomain.diameter[1] << " " << subdomain.diameter[2]
         << std::endl;
    fout.close();
}
}  // namespace io
}  // namespace mrmd