#include "action/LangevinThermostat.hpp"

#include <gtest/gtest.h>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = true;
    idx_t outputInterval = 10;

    idx_t nsteps = 21;
    real_t dt = 0.001_r;
    real_t temperature = 1.12_r;
    real_t gamma = 1_r / dt;

    real_t Lx = 10_r;
    real_t numParticles = 100000;

    real_t initialMaxVelocity = 10_r;
};

data::Particles fillDomainWithParticlesSC(const data::Subdomain& subdomain,
                                          const idx_t& numParticles,
                                          const real_t& maxVelocity)
{
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    data::Particles particles(numParticles);

    auto pos = particles.getPos();
    auto vel = particles.getVel();
    auto mass = particles.getMass();

    auto policy = Kokkos::RangePolicy<>(0, numParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto randGen = RNG.get_state();
        pos(idx, 0) = randGen.drand() * subdomain.diameter[0] + subdomain.minCorner[0];
        pos(idx, 1) = randGen.drand() * subdomain.diameter[1] + subdomain.minCorner[1];
        pos(idx, 2) = randGen.drand() * subdomain.diameter[2] + subdomain.minCorner[2];

        vel(idx, 0) = (randGen.drand() - 0.5_r) * maxVelocity;
        vel(idx, 1) = (randGen.drand() - 0.5_r) * maxVelocity;
        vel(idx, 2) = (randGen.drand() - 0.5_r) * maxVelocity;
        RNG.free_state(randGen);

        mass(idx) = 1_r;
    };
    Kokkos::parallel_for("fillDomainWithParticlesSC", policy, kernel);

    particles.numLocalParticles = numParticles;
    particles.numGhostParticles = 0;
    return particles;
}

TEST(LangevinThermostat, Integration)
{
    Config config;
    auto subdomain = data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Lx, config.Lx}, 1_r);
    auto particles =
        fillDomainWithParticlesSC(subdomain, config.numParticles, config.initialMaxVelocity);

    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    for (auto step = 0; step < config.nsteps; ++step)
    {
        action::VelocityVerlet::preForceIntegrate(particles, config.dt);

        auto force = particles.getForce();
        Cabana::deep_copy(force, 0_r);

        langevinThermostat.apply(particles);

        action::VelocityVerlet::postForceIntegrate(particles, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto Ek = analysis::getKineticEnergy(particles);
            auto T = (2_r / (3_r * real_c(particles.numLocalParticles))) * Ek;
            std::cout << "temperature: " << T << std::endl;
        }
    }
    auto Ek = analysis::getKineticEnergy(particles);
    auto T = (2_r / (3_r * real_c(particles.numLocalParticles))) * Ek;
    EXPECT_NEAR(T, config.temperature, 0.01_r);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    return RUN_ALL_TESTS();
}