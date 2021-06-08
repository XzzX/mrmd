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
    constexpr double nsteps = 100;
    constexpr double rc = 2.5;
    constexpr double skin = 0.3;
    constexpr double dt = 0.005;

    auto subdomain = Subdomain({0_r, 0_r, 0_r}, {33.8585, 33.8585, 33.8585}, rc + skin);
    Kokkos::Timer timer;
    auto particles = loadParticles("positions.txt");
    CHECK_EQUAL(particles.numLocalParticles, 32768);
    std::cout << "load particles: " << timer.seconds() << std::endl;

    double cell_ratio = 0.5_r;
    using ListType = Cabana::VerletList<Kokkos::HostSpace,
                                        Cabana::HalfNeighborTag,
                                        Cabana::VerletLayoutCSR,
                                        Cabana::TeamOpTag>;

    std::cout << "T: " << getTemperature(particles) << std::endl;

    for (auto i = 0; i < nsteps; ++i)
    {
        particles.numGhostParticles = 0;
        auto force = particles.getForce();
        Cabana::deep_copy(force, 0_r);
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

        auto positions = particles.getPos();

        ListType verlet_list(positions,
                             0,
                             particles.numLocalParticles,
                             rc + skin,
                             cell_ratio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data());

        Integrator integrator(dt);
        integrator.preForceIntegrate(particles);
        Kokkos::fence();

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

        std::cout << i << ": " << timer.seconds() << "s" << std::endl;
        std::cout << "T: " << getTemperature(particles) << std::endl;

        auto countPairs = KOKKOS_LAMBDA(const int i, const int j, idx_t& sum) { ++sum; };

        idx_t numPairs = 0;
        real_t totalEnergy = 0_r;
        Cabana::neighbor_parallel_reduce(
            Kokkos::RangePolicy<Kokkos::Serial>(0, particles.numLocalParticles),
            LennardJonesEnergy(particles, rc, 1_r, 1_r),
            verlet_list,
            Cabana::FirstNeighborsTag(),
            Cabana::SerialOpTag(),
            totalEnergy,
            "LennardJonesEnergy");
        std::cout << "E: " << totalEnergy << std::endl;
    }
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    LJ();

    return EXIT_SUCCESS;
}