#include "action/LennardJones.hpp"

#include <Kokkos_Core.hpp>
#include <fstream>

#include "Cabana_NeighborList.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/Temperature.hpp"
#include "checks.hpp"
#include "communication/AccumulateForce.hpp"
#include "communication/GhostExchange.hpp"
#include "communication/PeriodicMapping.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"

Particles loadParticles(const std::string& filename)
{
    Particles p(100000);
    auto d_AoSoA = p.getAoSoA();
    auto h_AoSoA = Cabana::create_mirror_view(Kokkos::HostSpace(), d_AoSoA);
    auto h_pos = Cabana::slice<Particles::POS>(h_AoSoA);

    std::ifstream fin(filename);

    idx_t idx = 0;
    while (!fin.eof())
    {
        double x, y, z;
        fin >> x >> y >> z;
        if (fin.eof()) break;
        h_pos(idx, 0) = x;
        h_pos(idx, 1) = y;
        h_pos(idx, 2) = z;
        ++idx;
    }

    fin.close();

    Cabana::deep_copy(d_AoSoA, h_AoSoA);

    p.numLocalParticles = idx;

    return p;
}

void LJ()
{
    constexpr bool bOutput = false;

    constexpr idx_t nsteps = 201;
    constexpr real_t rc = 2.5;
    constexpr real_t skin = 0.3;
    constexpr real_t dt = 0.005;

    constexpr real_t Lx = 33.8585;
    auto subdomain = Subdomain({0_r, 0_r, 0_r}, {Lx, Lx, Lx}, rc + skin);
    auto particles = loadParticles("positions.txt");
    CHECK_EQUAL(particles.numLocalParticles, 32768);

    double cell_ratio = 0.5_r;
    using ListType = Cabana::VerletList<Kokkos::HostSpace,
                                        Cabana::HalfNeighborTag,
                                        Cabana::VerletLayoutCSR,
                                        Cabana::TeamOpTag>;

    VelocityVerlet integrator(dt);
    PeriodicMapping periodicMapping(subdomain);
    GhostExchange ghostExchange(subdomain);
    LennardJones LJ(rc, 1_r, 1_r);
    AccumulateForce accumulateForce;
    Kokkos::Timer timer;
    for (auto i = 0; i < nsteps; ++i)
    {
        particles.removeGhostParticles();
        auto ghost = particles.getGhost();
        Cabana::deep_copy(ghost, idx_c(-1));

        integrator.preForceIntegrate(particles);
        Kokkos::fence();

        periodicMapping.mapIntoDomain(particles);
        Kokkos::fence();

        ghostExchange.exchangeGhostsXYZ(particles);
        particles.resize(particles.numLocalParticles + particles.numGhostParticles);

        ListType verlet_list(particles.getPos(),
                             0,
                             particles.numLocalParticles,
                             rc + skin,
                             cell_ratio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data());

        auto force = particles.getForce();
        Cabana::deep_copy(force, 0_r);

        LJ.applyForces(particles, verlet_list);
        Kokkos::fence();

        accumulateForce.ghostToReal(particles);
        Kokkos::fence();

        integrator.postForceIntegrate(particles);
        Kokkos::fence();

        if (bOutput && (i%100 == 0))
        {
            auto E0 = LJ.computeEnergy(particles, verlet_list);
            auto T = getTemperature(particles);
            auto Ek = (3.0 / 2.0) * particles.numLocalParticles * T;
            std::cout << "T : " << std::setw(10) << T << " | ";
            std::cout << "Ek: " << std::setw(10) << Ek << " | ";
            std::cout << "E0: " << std::setw(10) << E0 << " | ";
            std::cout << "E : " << std::setw(10) << E0 + Ek << " | ";
            std::cout << "Nlocal : " << std::setw(10) << particles.numLocalParticles << " | ";
            std::cout << "Nghost : " << std::setw(10) << particles.numGhostParticles << std::endl;
        }
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;

    auto cores = std::getenv("OMP_NUM_THREADS") != nullptr ? std::string(std::getenv("OMP_NUM_THREADS")) : std::string("0");

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << particles.numLocalParticles << ", " << nsteps << std::endl;
    fout.close();

    // dumpCSV("particles_" + std::to_string(i) + ".csv", particles);

    ListType verlet_list(particles.getPos(),
                         0,
                         particles.numLocalParticles,
                         rc + skin,
                         cell_ratio,
                         subdomain.minGhostCorner.data(),
                         subdomain.maxGhostCorner.data());
    auto E0 = LJ.computeEnergy(particles, verlet_list);
    auto T = getTemperature(particles);

//    CHECK_LESS(E0, -162000_r);
//    CHECK_GREATER(E0, -163000_r);
//
//    CHECK_LESS(T, 1.43_r);
//    CHECK_GREATER(T, 1.41_r);
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    LJ();

    return EXIT_SUCCESS;
}