#include "RestoreTXT.hpp"

#include <fstream>

namespace mrmd
{
namespace io
{
data::Particles restoreParticles(const std::string& filename)
{
    constexpr idx_t numInitiallyAllocatedParticles = 100000;
    data::Particles p(numInitiallyAllocatedParticles);
    auto d_AoSoA = p.getAoSoA();
    auto h_AoSoA = Cabana::create_mirror_view(Kokkos::HostSpace(), d_AoSoA);
    auto h_pos = Cabana::slice<data::Particles::POS>(h_AoSoA);

    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    idx_t idx = 0;
    while (!fin.eof())
    {
        double x;
        double y;
        double z;
        fin >> x >> y >> z;
        if (fin.eof()) break;
        if (std::isnan(x) || std::isnan(y) || std::isnan(z))
        {
            std::cout << "invalid position: " << x << " " << y << " " << z << std::endl;
            exit(EXIT_FAILURE);
        }
        h_pos(idx, 0) = x;
        h_pos(idx, 1) = y;
        h_pos(idx, 2) = z;
        ++idx;
    }

    fin.close();

    Cabana::deep_copy(d_AoSoA, h_AoSoA);
    p.numLocalParticles = idx;
    p.resize(p.numLocalParticles);

    auto vel = p.getVel();
    auto force = p.getForce();
    Cabana::deep_copy(vel, 0_r);
    Cabana::deep_copy(force, 0_r);

    return p;
}
}  // namespace io
}  // namespace mrmd