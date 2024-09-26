// Copyright 2024 Sebastian Eibl
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    real_t dt = 0.001_r;
    real_t temperature = 1.12_r;
    real_t gamma = 1_r / dt;

    real_t Lx = 10_r;
    idx_t numAtoms = 100000;

    real_t initialMaxVelocity = 10_r;
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

        vel(idx, 0) = (randGen.drand() - 0.5_r) * maxVelocity;
        vel(idx, 1) = (randGen.drand() - 0.5_r) * maxVelocity;
        vel(idx, 2) = (randGen.drand() - 0.5_r) * maxVelocity;
        RNG.free_state(randGen);

        mass(idx) = 1_r;
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

TEST(Integration, LangevinThermostat)
{
    Config config;
    auto subdomain = data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Lx, config.Lx}, 1_r);
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numAtoms, config.initialMaxVelocity);

    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    for (auto step = 0; step < config.nsteps; ++step)
    {
        action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        langevinThermostat.apply(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto T = (2_r / 3_r) * Ek;
            std::cout << "temperature: " << T << std::endl;
        }
    }
    auto Ek = analysis::getMeanKineticEnergy(atoms);
    auto T = (2_r / 3_r) * Ek;
    EXPECT_NEAR(T, config.temperature, 0.01_r);
}

TEST(Integration, VelocityVerletLangevinThermostat)
{
    Config config;
    auto subdomain = data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Lx, config.Lx}, 1_r);
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numAtoms, config.initialMaxVelocity);

    action::VelocityVerletLangevinThermostat vv(1e5_r, config.temperature);
    for (auto step = 0; step < config.nsteps; ++step)
    {
        vv.preForceIntegrate(atoms, config.dt);
        vv.postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto T = (2_r / 3_r) * Ek;
            std::cout << "temperature: " << T << std::endl;
        }
    }
    auto Ek = analysis::getMeanKineticEnergy(atoms);
    auto T = (2_r / 3_r) * Ek;
    EXPECT_NEAR(T, config.temperature, 0.01_r);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    return RUN_ALL_TESTS();
}