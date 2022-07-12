#include "action/LangevinThermostat.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "action/VelocityVerlet.hpp"
#include "action/VelocityVerletLangevinThermostat.hpp"
#include "analysis/KineticEnergy.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = true;
    idx_t outputInterval = 10;

    idx_t nsteps = 21;
    real_t dt = real_t(0.001);
    real_t temperature = real_t(1.12);
    real_t gamma = real_t(1) / dt;

    real_t Lx = real_t(10);
    idx_t numAtoms = 100000;

    real_t initialMaxVelocity = real_t(10);
};

data::Atoms fillDomainWithAtomsSC(const data::Subdomain& subdomain,
                                  const idx_t& numAtoms,
                                  const real_t& maxVelocity)
{
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    data::Atoms atoms(numAtoms);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto randGen = RNG.get_state();
        pos(idx, 0) = randGen.drand() * subdomain.diameter[0] + subdomain.minCorner[0];
        pos(idx, 1) = randGen.drand() * subdomain.diameter[1] + subdomain.minCorner[1];
        pos(idx, 2) = randGen.drand() * subdomain.diameter[2] + subdomain.minCorner[2];

        vel(idx, 0) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        vel(idx, 1) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        vel(idx, 2) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        RNG.free_state(randGen);

        mass(idx) = real_t(1);
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

TEST(Integration, LangevinThermostat)
{
    Config config;
    auto subdomain = data::Subdomain(
        {real_t(0), real_t(0), real_t(0)}, {config.Lx, config.Lx, config.Lx}, real_t(1));
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numAtoms, config.initialMaxVelocity);

    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    for (auto step = 0; step < config.nsteps; ++step)
    {
        action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        auto force = atoms.getForce();
        Cabana::deep_copy(force, real_t(0));

        langevinThermostat.apply(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto T = (real_t(2) / real_t(3)) * Ek;
            std::cout << "temperature: " << T << std::endl;
        }
    }
    auto Ek = analysis::getMeanKineticEnergy(atoms);
    auto T = (real_t(2) / real_t(3)) * Ek;
    EXPECT_NEAR(T, config.temperature, real_t(0.01));
}

TEST(Integration, VelocityVerletLangevinThermostat)
{
    Config config;
    auto subdomain = data::Subdomain(
        {real_t(0), real_t(0), real_t(0)}, {config.Lx, config.Lx, config.Lx}, real_t(1));
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numAtoms, config.initialMaxVelocity);

    action::VelocityVerletLangevinThermostat vv(real_t(1e5), config.temperature);
    for (auto step = 0; step < config.nsteps; ++step)
    {
        vv.preForceIntegrate(atoms, config.dt);
        vv.postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto T = (real_t(2) / real_t(3)) * Ek;
            std::cout << "temperature: " << T << std::endl;
        }
    }
    auto Ek = analysis::getMeanKineticEnergy(atoms);
    auto T = (real_t(2) / real_t(3)) * Ek;
    EXPECT_NEAR(T, config.temperature, real_t(0.01));
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    return RUN_ALL_TESTS();
}