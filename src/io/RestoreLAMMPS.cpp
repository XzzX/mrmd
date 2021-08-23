#include "RestoreLAMMPS.hpp"

#include <cassert>
#include <fstream>

namespace mrmd
{
namespace io
{
void restoreLAMMPS(const std::string& filename, data::Particles& atoms, data::Molecules& molecules)
{
    std::string tmp;
    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    fin >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp >> tmp;

    idx_t numParticles;
    fin >> numParticles;
    atoms.resize(2 * numParticles);
    auto d_Atoms = atoms.getAoSoA();
    auto h_Atoms = Cabana::create_mirror_view(Kokkos::HostSpace(), d_Atoms);
    auto h_pos = Cabana::slice<data::Particles::POS>(h_Atoms);
    auto h_vel = Cabana::slice<data::Particles::VEL>(h_Atoms);
    auto h_relativeMass = Cabana::slice<data::Particles::RELATIVE_MASS>(h_Atoms);

    molecules.resize(2 * numParticles);
    auto d_Molecules = molecules.getAoSoA();
    auto h_Molecules = Cabana::create_mirror_view(Kokkos::HostSpace(), d_Molecules);
    auto h_AtomsOffset = Cabana::slice<data::Molecules::ATOMS_OFFSET>(h_Molecules);
    auto h_NumAtoms = Cabana::slice<data::Molecules::NUM_ATOMS>(h_Molecules);

    fin >> tmp >> tmp >> tmp >> tmp >> tmp;
    fin >> tmp >> tmp >> tmp >> tmp >> tmp;
    fin >> tmp >> tmp >> tmp >> tmp >> tmp;
    fin >> tmp >> tmp >> tmp >> tmp >> tmp;
    fin >> tmp >> tmp >> tmp;

    idx_t idx = 0;
    while (!fin.eof())
    {
        idx_t particleIndexInFile;

        double posX;
        double posY;
        double posZ;

        double velX;
        double velY;
        double velZ;

        fin >> particleIndexInFile;
        if (fin.eof()) break;
        fin >> tmp >> tmp >> posX >> posY >> posZ >> velX >> velY >> velZ;
        assert(particleIndexInFile == idx + 1);
        if (std::isnan(posX) || std::isnan(posY) || std::isnan(posZ))
        {
            std::cout << "invalid position: " << posX << " " << posY << " " << posZ << std::endl;
            exit(EXIT_FAILURE);
        }
        h_pos(idx, 0) = posX;
        h_pos(idx, 1) = posY;
        h_pos(idx, 2) = posZ;

        h_vel(idx, 0) = velX;
        h_vel(idx, 1) = velY;
        h_vel(idx, 2) = velZ;

        h_relativeMass(idx) = 1_r;

        h_AtomsOffset(idx) = idx;
        h_NumAtoms(idx) = 1;

        ++idx;
    }

    fin.close();

    Cabana::deep_copy(d_Atoms, h_Atoms);
    Cabana::deep_copy(d_Molecules, h_Molecules);
    assert(idx == numParticles);
    atoms.numLocalParticles = idx;
    molecules.numLocalMolecules = idx;

    auto force = atoms.getForce();
    Cabana::deep_copy(force, 0_r);
}
}  // namespace io
}  // namespace mrmd