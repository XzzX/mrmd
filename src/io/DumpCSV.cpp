#include "io/DumpCSV.hpp"

#include <fstream>

#include "data/Particles.hpp"

namespace mrmd
{
namespace io
{
void dumpCSV(const std::string& filename, data::Particles& particles, bool dumpGhosts)
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
    fout << "pos_x, pos_y, pos_z, vel_x, vel_y, vel_z" << std::endl;
    auto lastParticleIdx =
        particles.numLocalParticles + (dumpGhosts ? particles.numGhostParticles : 0);
    for (idx_t idx = 0; idx < lastParticleIdx; ++idx)
    {
        fout << pos(idx, 0) << ", " << pos(idx, 1) << ", " << pos(idx, 2) << ", " << vel(idx, 0)
             << ", " << vel(idx, 1) << ", " << vel(idx, 2) << std::endl;
    }
    fout.close();
}
}  // namespace io
}  // namespace mrmd