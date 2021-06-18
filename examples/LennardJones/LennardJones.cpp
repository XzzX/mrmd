#include "LennardJones.hpp"

#include <Kokkos_Core.hpp>
#include <fstream>

#include "AccumulateForce.hpp"
#include "Cabana_NeighborList.hpp"
#include "HaloExchange.hpp"
#include "Integrator.hpp"
#include "PeriodicMapping.hpp"
#include "Subdomain.hpp"
#include "Temperature.hpp"
#include "checks.hpp"

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
    constexpr double nsteps = 1001;
    constexpr double rc = 2.5;
    constexpr double skin = 0.3;
    constexpr double dt = 0.005;

    auto subdomain = Subdomain({0_r, 0_r, 0_r}, {33.8585, 33.8585, 33.8585}, 2_r * (rc + skin));
    Kokkos::Timer timer;
    auto particles = loadParticles("positions.txt");
    CHECK_EQUAL(particles.numLocalParticles, 32768);
    std::cout << "load particles: " << timer.seconds() << std::endl;

    double cell_ratio = 0.5_r;
    using ListType = Cabana::VerletList<Kokkos::HostSpace,
                                        Cabana::HalfNeighborTag,
                                        Cabana::VerletLayoutCSR,
                                        Cabana::TeamOpTag>;

    for (auto i = 0; i < nsteps; ++i)
    {
        particles.numGhostParticles = 0;

        auto ghost = particles.getGhost();
        Cabana::deep_copy(ghost, idx_c(-1));

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

        ListType verlet_list(particles.getPos(),
                             0,
                             particles.numLocalParticles,
                             rc + skin,
                             cell_ratio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data());

        Integrator integrator(dt);
        integrator.preForceIntegrate(particles);
        Kokkos::fence();

        auto force = particles.getForce();
        Cabana::deep_copy(force, 0_r);
        Cabana::neighbor_parallel_for(
            Kokkos::RangePolicy<Kokkos::Serial>(0, particles.numLocalParticles),
            LennardJones(particles, rc, 1_r, 1_r),
            verlet_list,
            Cabana::FirstNeighborsTag(),
            Cabana::SerialOpTag(),
            "LennardJonesForce");
        Kokkos::fence();

        Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(
                                 particles.numLocalParticles,
                                 particles.numLocalParticles + particles.numGhostParticles),
                             AccumulateForce(particles));

        integrator.postForceIntegrate(particles);
        Kokkos::fence();

        if (i % 100 == 0)
        {
            auto countPairs = KOKKOS_LAMBDA(const int i, const int j, idx_t& sum) { ++sum; };

            idx_t numPairs = 0;
            real_t E0 = 0_r;
            Cabana::neighbor_parallel_reduce(
                Kokkos::RangePolicy<Kokkos::Serial>(0, particles.numLocalParticles),
                LennardJonesEnergy(particles, rc, 1_r, 1_r),
                verlet_list,
                Cabana::FirstNeighborsTag(),
                Cabana::SerialOpTag(),
                E0,
                "LennardJonesEnergy");

            //        std::cout << i << ": " << timer.seconds() << "s" << std::endl;
            auto T = getTemperature(particles);
            auto Ek = (3.0 / 2.0) * particles.numLocalParticles * T;
            std::cout << "T : " << std::setw(10) << T << " | ";
            std::cout << "Ek: " << std::setw(10) << Ek << " | ";
            std::cout << "E0: " << std::setw(10) << E0 << " | ";
            std::cout << "E : " << std::setw(10) << E0 + Ek << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    LJ();

    return EXIT_SUCCESS;
}