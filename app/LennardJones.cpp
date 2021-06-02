#include <Kokkos_Core.hpp>

#include <fstream>

#include "Cabana_NeighborList.hpp"
#include "HaloExchange.hpp"
#include "Integrator.hpp"
#include "LennardJones.hpp"
#include "Subdomain.hpp"
#include "checks.hpp"

Particles loadParticles(const std::string filename)
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

size_t countWithinCutoff(Particles& particles, const real_t& cutoff, const double* box)
{
    auto rcSqr = cutoff * cutoff;
    auto pos = particles.getPos();

    size_t count = 0;
    for (auto idx = 0; idx < particles.size(); ++idx)
        for (auto jdx = idx + 1; jdx < particles.size(); ++jdx)
        {
            auto dx = std::abs(pos(idx, 0) - pos(jdx, 0));
            if (dx > box[0] * 0.5_r) dx -= box[0];
            auto dy = std::abs(pos(idx, 1) - pos(jdx, 1));
            if (dy > box[1] * 0.5_r) dy -= box[1];
            auto dz = std::abs(pos(idx, 2) - pos(jdx, 2));
            if (dz > box[2] * 0.5_r) dz -= box[2];
            auto distSqr = dx * dx + dy * dy + dz * dz;
            if (distSqr < rcSqr)
            {
                ++count;
            }
        }
    return count;
}

void LJ()
{
    auto subdomain = Subdomain({0_r, 0_r, 0_r}, {33.8585, 33.8585, 33.8585}, 2.5_r);
    Kokkos::Timer timer;
    auto particles = loadParticles("positions.txt");
    CHECK_EQUAL(particles.numLocalParticles, 32768);
    std::cout << "load particles: " << timer.seconds() << std::endl;

    constexpr double nsteps = 100;
    constexpr double rc = 2.5;
    constexpr double skin = 0.3;
    constexpr double dt = 0.005;

    double neighborhood_radius = 2.8_r;
    double cell_ratio = 0.5_r;
    using ListType = Cabana::VerletList<Kokkos::HostSpace,
                                        Cabana::HalfNeighborTag,
                                        Cabana::VerletLayoutCSR,
                                        Cabana::TeamOpTag>;
    auto positions = particles.getPos();

    std::cout << "bf neighbors: "
              //              << countWithinCutoff(particles, neighborhood_radius,
              //              subdomain.diameter.data())
              << std::endl;
    std::cout << "brute force: " << timer.seconds() << std::endl;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::Serial>(0, particles.numLocalParticles),
        HaloExchange(particles, subdomain));
    std::cout << "local particles: " << particles.numLocalParticles << std::endl;
    std::cout << "ghost particles: " << particles.numGhostParticles << std::endl;

    ListType verlet_list(positions,
                         0,
                         particles.numLocalParticles,
                         neighborhood_radius,
                         cell_ratio,
                         subdomain.minGhostCorner.data(),
                         subdomain.maxGhostCorner.data());
    size_t sum = 0;
    Kokkos::parallel_reduce(
        verlet_list._data.counts.size(),
        KOKKOS_LAMBDA(const int idx, size_t& count) { count += verlet_list._data.counts(idx); },
        sum);
    std::cout << "found " << sum << " neighbors" << std::endl;
    std::cout << "create verlet list: " << timer.seconds() << std::endl;

    for (auto i = 0; i < nsteps; ++i)
    {
        Integrator integrator(dt);
        integrator.preForceIntegrate(particles);
        Kokkos::fence();
        //        std::cout << "pre force integrate: " << timer.seconds() <<
        //        std::endl;

        Kokkos::RangePolicy<Kokkos::Serial> policy(
            0, particles.numLocalParticles + particles.numGhostParticles);
        Cabana::neighbor_parallel_for(policy,
                                      LennardJones(particles, rc, 1_r, 1_r),
                                      verlet_list,
                                      Cabana::FirstNeighborsTag(),
                                      Cabana::SerialOpTag(),
                                      "LennardJones");
        Kokkos::fence();
        //        std::cout << "lennard jones: " << timer.seconds() <<
        //        std::endl;

        integrator.postForceIntegrate(particles);
        Kokkos::fence();
        //        std::cout << "post force integrate: " << timer.seconds() <<
        //        std::endl;
    }
    std::cout << "finished: " << timer.seconds() << std::endl;
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    LJ();

    return EXIT_SUCCESS;
}