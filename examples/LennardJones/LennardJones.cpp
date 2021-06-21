#include "action/LennardJones.hpp"

#include <Kokkos_Core.hpp>
#include <fstream>

#include "Cabana_NeighborList.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/Temperature.hpp"
#include "checks.hpp"
#include "communication/AccumulateForce.hpp"
#include "communication/HaloExchange.hpp"
#include "communication/PeriodicMapping.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"

Particles loadParticles(const std::string& filename)
{
    Particles p;
    auto pos = p.getPos();

    std::ifstream fin(filename);

    idx_t idx = 0;
    while (!fin.eof())
    {
        double x, y, z;
        fin >> x >> y >> z;
        if (fin.eof()) break;
        pos(idx, 0) = x;
        pos(idx, 1) = y;
        pos(idx, 2) = z;
        ++idx;
    }

    fin.close();

    p.numLocalParticles = idx;

    return p;
}

void LJ()
{
    constexpr double nsteps = 2001;
    constexpr double rc = 2.5;
    constexpr double skin = 0.3;
    constexpr double dt = 0.005;

    constexpr real_t Lx = 33.8585;
    auto subdomain = Subdomain({0_r, 0_r, 0_r}, {Lx, Lx, Lx}, 2_r * (rc + skin));
    Kokkos::Timer timer;
    auto particles = loadParticles("positions.txt");
    CHECK_EQUAL(particles.numLocalParticles, 32768);
    std::cout << "load particles: " << timer.seconds() << std::endl;

    double cell_ratio = 0.5_r;
    using ListType = Cabana::VerletList<Kokkos::HostSpace,
                                        Cabana::HalfNeighborTag,
                                        Cabana::VerletLayoutCSR,
                                        Cabana::TeamOpTag>;

    VelocityVerlet integrator(dt);
    LennardJones LJ(rc, 1_r, 1_r);
    for (auto i = 0; i < nsteps; ++i)
    {
        particles.numGhostParticles = 0;
        auto ghost = particles.getGhost();
        Cabana::deep_copy(ghost, idx_c(-1));

        integrator.preForceIntegrate(particles);
        Kokkos::fence();

        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, particles.numLocalParticles),
                             PeriodicMapping(particles, subdomain));

        auto haloExchange = HaloExchange(particles, subdomain);
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagX>(
                                 0, particles.numLocalParticles + particles.numGhostParticles),
                             haloExchange);
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagY>(
                                 0, particles.numLocalParticles + particles.numGhostParticles),
                             haloExchange);
        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial, HaloExchange::TagZ>(
                                 0, particles.numLocalParticles + particles.numGhostParticles),
                             haloExchange);
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

        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(
                                 particles.numLocalParticles,
                                 particles.numLocalParticles + particles.numGhostParticles),
                             AccumulateForce(particles));

        integrator.postForceIntegrate(particles);
        Kokkos::fence();

        if (i % 100 == 0)
        {
            dumpCSV("particles_" + std::to_string(i) + ".csv", particles);
            auto countPairs = KOKKOS_LAMBDA(const int i, const int j, idx_t& sum) { ++sum; };

            idx_t numPairs = 0;
            auto E0 = LJ.computeEnergy(particles, verlet_list);

            //        std::cout << i << ": " << timer.seconds() << "s" << std::endl;
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
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    LJ();

    return EXIT_SUCCESS;
}