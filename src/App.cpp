#include <fstream>

#include "Cabana_NeighborList.hpp"
#include "Integrator.hpp"
#include "LennardJones.hpp"
#include "checks.hpp"

Particles loadParticles(const std::string filename)
{
    Particles p;
    auto pos = p.getPos();

    std::ifstream fin(filename);

    size_t idx = 0;
    while (!fin.eof())
    {
        double x, y, z;
        fin >> x >> y >> z;
        if (fin.eof()) break;
        pos(0, idx) = x;
        pos(1, idx) = y;
        pos(2, idx) = z;
        ++idx;
    }

    fin.close();

    p.resize(idx);

    return p;
}

void LJ()
{
    Kokkos::Timer timer;
    auto particles = loadParticles("positions.txt");
    CHECK_EQUAL(particles.size(), 32768);
    std::cout << "load particles: " << timer.seconds() << std::endl;

    constexpr double nsteps = 1000;
    constexpr double rc = 2.5;
    constexpr double skin = 0.3;
    constexpr double dt = 0.005;

    double grid_min[3] = {0.0, 0.0, 0.0};
    double grid_max[3] = {33.8585, 33.8585, 33.8585};

    double neighborhood_radius = 2.8_r;
    double cell_ratio = 0.5_r;
    using ListType = Cabana::VerletList<Kokkos::HostSpace,
                                        Cabana::HalfNeighborTag,
                                        Cabana::VerletLayoutCSR,
                                        Cabana::TeamOpTag>;
    auto positions = particles.getPos();
    CHECK_EQUAL(positions.size(), 32768);

    ListType verlet_list(
        positions, 0, positions.size(), neighborhood_radius, cell_ratio, grid_min, grid_max);
    size_t sum = 0;
    Kokkos::parallel_reduce(
        verlet_list._data.counts.size(),
        KOKKOS_LAMBDA(const int idx, size_t& count) { count += verlet_list._data.counts(idx); },
        sum);
    std::cout << "found " << sum << " neighbors" << std::endl;
    std::cout << "create verlet list: " << timer.seconds() << std::endl;

    Integrator integrator(dt);
    integrator.preForceIntegrate(particles);
    Kokkos::fence();
    std::cout << "pre force integrate: " << timer.seconds() << std::endl;

    Kokkos::RangePolicy<Kokkos::Serial> policy(0, particles.size());
    Cabana::neighbor_parallel_for(policy,
                                  LennardJones(particles, rc, 1_r, 1_r),
                                  verlet_list,
                                  Cabana::FirstNeighborsTag(),
                                  Cabana::SerialOpTag(),
                                  "LennardJones");
    Kokkos::fence();
    std::cout << "lennard jones: " << timer.seconds() << std::endl;

    integrator.postForceIntegrate(particles);
    Kokkos::fence();
    std::cout << "post force integrate: " << timer.seconds() << std::endl;
}