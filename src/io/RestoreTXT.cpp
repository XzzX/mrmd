#include "RestoreTXT.hpp"

#include <fstream>

namespace mrmd
{
namespace io
{
data::Atoms restoreAtoms(const std::string& filename)
{
    constexpr idx_t numInitiallyAllocatedAtoms = 100000;
    data::Atoms p(numInitiallyAllocatedAtoms);
    auto d_AoSoA = p.getAoSoA();
    auto h_AoSoA = Cabana::create_mirror_view(Kokkos::HostSpace(), d_AoSoA);
    auto h_pos = Cabana::slice<data::Atoms::POS>(h_AoSoA);
    auto h_mass = Cabana::slice<data::Atoms::MASS>(h_AoSoA);
    auto h_type = Cabana::slice<data::Atoms::TYPE>(h_AoSoA);

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
        h_mass(idx) = 1_r;
        h_type(idx) = 0;
        ++idx;
    }

    fin.close();

    Cabana::deep_copy(d_AoSoA, h_AoSoA);
    p.numLocalAtoms = idx;
    p.resize(p.numLocalAtoms);

    auto vel = p.getVel();
    auto force = p.getForce();
    Cabana::deep_copy(vel, 0_r);
    Cabana::deep_copy(force, 0_r);

    return p;
}
}  // namespace io
}  // namespace mrmd