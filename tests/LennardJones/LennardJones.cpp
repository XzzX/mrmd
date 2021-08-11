#include "action/LennardJones.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <fstream>

#include "Cabana_NeighborList.hpp"
#include "action/VelocityVerlet.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Subdomain.hpp"
#include "io/RestoreTXT.hpp"

namespace mrmd
{
/// reference values from espressopp simulation
/// number of real particles
constexpr idx_t ESPP_REAL = 32768;
/// number of ghost particles
constexpr idx_t ESPP_GHOST = 22104;
/// number of pairs in the neighbor list
constexpr idx_t ESPP_NEIGHBORS = 1310403;
/// initial lennard jones energy
constexpr real_t ESPP_INITIAL_ENERGY = -94795.927_r;

idx_t countWithinCutoff(data::Particles& particles,
                        const real_t& cutoff,
                        const std::array<real_t, 3>& box,
                        const bool periodic)
{
    auto rcSqr = cutoff * cutoff;
    auto pos = particles.getPos();

    idx_t count = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, particles.numLocalParticles),
        KOKKOS_LAMBDA(const idx_t idx, idx_t& sum)
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

size_t countPairs(VerletList& vl)
{
    size_t vlParticlePairs = 0;
    Kokkos::parallel_reduce(
        vl._data.counts.size(),
        KOKKOS_LAMBDA(const int idx, size_t& count) { count += vl._data.counts(idx); },
        vlParticlePairs);
    return vlParticlePairs;
}

TEST(LennardJones, ESPPComparison)
{
    constexpr double rc = 2.5;
    constexpr double skin = 0.3;
    constexpr double dt = 0.005;

    auto subdomain = data::Subdomain({0_r, 0_r, 0_r}, {33.8585, 33.8585, 33.8585}, rc + skin);
    Kokkos::Timer timer;
    auto particles = io::restoreParticles("positions.txt");
    EXPECT_EQ(particles.numLocalParticles, ESPP_REAL);
    std::cout << "load particles: " << timer.seconds() << std::endl;

    idx_t bfParticlePairs = 0;
    bfParticlePairs = countWithinCutoff(particles, rc + skin, subdomain.diameter, true);
    EXPECT_EQ(bfParticlePairs, ESPP_NEIGHBORS);
    std::cout << "brute force: " << timer.seconds() << std::endl;

    auto ghostExchange = communication::GhostLayer(subdomain);
    ghostExchange.createGhostParticles(particles);
    Kokkos::fence();
    EXPECT_EQ(particles.numLocalParticles, ESPP_REAL);
    EXPECT_EQ(particles.numGhostParticles, ESPP_GHOST);
    particles.resize(particles.numLocalParticles + particles.numGhostParticles);
    std::cout << "halo exchange: " << timer.seconds() << std::endl;

    bfParticlePairs = countWithinCutoff(particles, rc + skin, subdomain.diameter, false);
    EXPECT_EQ(bfParticlePairs, 1426948);
    std::cout << "brute force: " << timer.seconds() << std::endl;

    double cell_ratio = 1.0_r;
    VerletList verlet_list(particles.getPos(),
                           0,
                           particles.numLocalParticles,
                           rc + skin,
                           cell_ratio,
                           subdomain.minGhostCorner.data(),
                           subdomain.maxGhostCorner.data());

    size_t vlParticlePairs = countPairs(verlet_list);
    EXPECT_EQ(vlParticlePairs, ESPP_NEIGHBORS);
    std::cout << "create verlet list: " << timer.seconds() << std::endl;

    action::LennardJones LJ(rc, 1_r, 1_r);
    real_t totalEnergy = LJ.computeEnergy(particles, verlet_list);
    EXPECT_FLOAT_EQ(totalEnergy, ESPP_INITIAL_ENERGY);
    std::cout << "starting energy: " << totalEnergy << std::endl;

    action::VelocityVerlet::preForceIntegrate(particles, dt);
    Kokkos::fence();
    std::cout << "pre force integrate: " << timer.seconds() << std::endl;

    LJ.applyForces(particles, verlet_list);
    Kokkos::fence();
    std::cout << "lennard jones: " << timer.seconds() << std::endl;

    action::VelocityVerlet::postForceIntegrate(particles, dt);
    Kokkos::fence();
    std::cout << "post force integrate: " << timer.seconds() << std::endl;

    std::cout << "finished: " << timer.seconds() << std::endl;
}

}  // namespace mrmd

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    return RUN_ALL_TESTS();
}
