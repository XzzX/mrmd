#include "LJ.hpp"
#include "checks.hpp"
#include "Integrator.hpp"

#include "Cabana_NeighborList.hpp"

#include <fstream>

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

    constexpr double nsteps      = 1000;
    constexpr double rc          = 2.5;
    constexpr double skin        = 0.3;
    constexpr double dt          = 0.005;

    double grid_min[3] = { 0.0, 0.0, 0.0 };
    double grid_max[3] = { 33.8585, 33.8585, 33.8585 };
    double grid_delta[3] = { rc, rc, rc };

    double neighborhood_radius = 2.5;
    double cell_ratio = 1.0;
    using ListAlgorithm = Cabana::FullNeighborTag;
    using ListType =
    Cabana::VerletList<Particles::DeviceType, ListAlgorithm, Cabana::VerletLayoutCSR,
        Cabana::TeamOpTag>;
    auto positions = particles.getPos();
    std::cout << positions.extent(0) << " | " << positions.extent(1) << " | " << positions.extent(2) << std::endl;
    CHECK_EQUAL(positions.size(), 32768);
//    ListType verlet_list( positions, 0, positions.size(), neighborhood_radius,
//                          cell_ratio, grid_min, grid_max );
    std::cout << "create verlet list: " << timer.seconds() << std::endl;

//    auto first_neighbor_kernel = KOKKOS_LAMBDA( const int i, const int j )
//    {
//      Kokkos::atomic_add( &slice_i( i ), slice_n( j ) );
//    };
//
//    Kokkos::RangePolicy<ExecutionSpace> policy( 0, aosoa.size() );
//
//    Cabana::neighbor_parallel_for( policy, first_neighbor_kernel, verlet_list,
//                                   Cabana::FirstNeighborsTag(),
//                                   Cabana::SerialOpTag(), "ex_1st_serial" );
//    Kokkos::fence();

    Integrator integrator(dt);
    integrator.preForceIntegrate(particles);
    integrator.postForceIntegrate(particles);
    std::cout << "integrate: " << timer.seconds() << std::endl;
}