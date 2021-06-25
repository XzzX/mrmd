#include "action/LennardJones.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <fstream>

#include "Cabana_NeighborList.hpp"
#include "action/VelocityVerlet.hpp"
#include "communication/GhostExchange.hpp"
#include "data/Subdomain.hpp"

/// reference values from espressopp simulation
/// number of real particles
constexpr idx_t ESPP_REAL = 32768;
/// number of ghost particles
constexpr idx_t ESPP_GHOST = 22104;
/// number of pairs in the neighbor list
constexpr idx_t ESPP_NEIGHBORS = 1310403;
/// initial lennard jones energy
constexpr real_t ESPP_INITIAL_ENERGY = -94795.927_r;

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
    auto ghost = p.getGhost();
    Cabana::deep_copy(ghost, -1);

    return p;
}

size_t countWithinCutoff(Particles& particles,
                         const real_t& cutoff,
                         const double* box,
                         const bool periodic)
{
    auto rcSqr = cutoff * cutoff;
    auto pos = particles.getPos();

    size_t count = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, particles.numLocalParticles),
        KOKKOS_LAMBDA(const idx_t idx, size_t& sum)
        {
            for (auto jdx = idx + 1;
                 jdx < particles.numLocalParticles + particles.numGhostParticles;
                 ++jdx)
            {
                auto dx = std::abs(pos(idx, 0) - pos(jdx, 0));
                if (periodic && (dx > box[0] * 0.5_r)) dx -= box[0];
                auto dy = std::abs(pos(idx, 1) - pos(jdx, 1));
                if (periodic && (dy > box[1] * 0.5_r)) dy -= box[1];
                auto dz = std::abs(pos(idx, 2) - pos(jdx, 2));
                if (periodic && (dz > box[2] * 0.5_r)) dz -= box[2];
                auto distSqr = dx * dx + dy * dy + dz * dz;
                if (distSqr < rcSqr)
                {
                    ++sum;
                }
            }
        },
        count);

    return count;
}

TEST(LennardJones, ESPPComparison)
{
    constexpr double rc = 2.5;
    constexpr double skin = 0.3;
    constexpr double dt = 0.005;

    auto subdomain = Subdomain({0_r, 0_r, 0_r}, {33.8585, 33.8585, 33.8585}, rc + skin);
    Kokkos::Timer timer;
    auto particles = loadParticles("positions.txt");
    EXPECT_EQ(particles.numLocalParticles, ESPP_REAL);
    std::cout << "load particles: " << timer.seconds() << std::endl;

    auto bfParticlePairs = 0;
    bfParticlePairs = countWithinCutoff(particles, rc + skin, subdomain.diameter.data(), true);
    EXPECT_EQ(bfParticlePairs, ESPP_NEIGHBORS);
    std::cout << "brute force: " << timer.seconds() << std::endl;

    auto ghostExchange = GhostExchange(subdomain);
    ghostExchange.exchangeGhostsXYZ(particles);
    Kokkos::fence();
    EXPECT_EQ(particles.numLocalParticles, ESPP_REAL);
    EXPECT_EQ(particles.numGhostParticles, ESPP_GHOST);
    particles.resize(particles.numLocalParticles + particles.numGhostParticles);
    std::cout << "halo exchange: " << timer.seconds() << std::endl;

    bfParticlePairs = countWithinCutoff(particles, rc + skin, subdomain.diameter.data(), false);
    EXPECT_EQ(bfParticlePairs, 1426948);
    std::cout << "brute force: " << timer.seconds() << std::endl;

    double cell_ratio = 1.0_r;
    using ListType = Cabana::VerletList<Kokkos::DefaultExecutionSpace::memory_space,
                                        Cabana::HalfNeighborTag,
                                        Cabana::VerletLayoutCSR,
                                        Cabana::TeamOpTag>;
    ListType verlet_list(particles.getPos(),
                         0,
                         particles.numLocalParticles,
                         rc + skin,
                         cell_ratio,
                         subdomain.minGhostCorner.data(),
                         subdomain.maxGhostCorner.data());
    size_t vlParticlePairs = 0;
    Kokkos::parallel_reduce(
        verlet_list._data.counts.size(),
        KOKKOS_LAMBDA(const int idx, size_t& count) { count += verlet_list._data.counts(idx); },
        vlParticlePairs);
    EXPECT_EQ(vlParticlePairs, ESPP_NEIGHBORS);
    std::cout << "create verlet list: " << timer.seconds() << std::endl;

    LennardJones LJ(rc, 1_r, 1_r);
    real_t totalEnergy = LJ.computeEnergy(particles, verlet_list);
    EXPECT_FLOAT_EQ(totalEnergy, ESPP_INITIAL_ENERGY);
    std::cout << "starting energy: " << totalEnergy << std::endl;

    VelocityVerlet integrator(dt);
    integrator.preForceIntegrate(particles);
    Kokkos::fence();
    std::cout << "pre force integrate: " << timer.seconds() << std::endl;

    LJ.applyForces(particles, verlet_list);
    Kokkos::fence();
    std::cout << "lennard jones: " << timer.seconds() << std::endl;

    integrator.postForceIntegrate(particles);
    Kokkos::fence();
    std::cout << "post force integrate: " << timer.seconds() << std::endl;

    std::cout << "finished: " << timer.seconds() << std::endl;
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    return RUN_ALL_TESTS();
}
