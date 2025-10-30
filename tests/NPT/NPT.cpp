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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/BerendsenBarostat.hpp"
#include "action/BerendsenThermostat.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"
#include "io/RestoreTXT.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/ExponentialMovingAverage.hpp"
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    static constexpr bool bOutput = true;
    static constexpr idx_t outputInterval = -1;

    static constexpr idx_t nsteps = 2001;
    static constexpr real_t rc = 2.5;
    static constexpr real_t skin = 0.3;
    static constexpr real_t neighborCutoff = rc + skin;
    static constexpr real_t sigma = 1_r;
    static constexpr real_t epsilon = 1_r;
    static constexpr real_t dt = 0.005;
    static constexpr real_t gamma = 0.1_r;

    static constexpr real_t Lx = 16.92926877476863_r;
    static constexpr real_t rho = 0.8442_r;

    static constexpr real_t cell_ratio = 1.0_r;

    static constexpr idx_t estimatedMaxNeighbors = 60;

    static constexpr real_t weightingFactor = 0.02_r;
};

data::Atoms fillDomainWithAtomsSC(const data::Subdomain &subdomain,
                                  const idx_t &numAtoms,
                                  const real_t &maxVelocity)
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

struct Input
{
    real_t targetTemperature;
    real_t targetPressure;
};

std::ostream &operator<<(std::ostream &os, const Input &input)
{
    os << "T: " << input.targetTemperature << " | "
       << "p: " << input.targetPressure;
    return os;
}

class NPT : public ::testing::TestWithParam<Input>
{
protected:
    // void SetUp() override {}
    // void TearDown() override {}

    data::Subdomain subdomain = {{0_r, 0_r, 0_r}, {Config::Lx, Config::Lx, Config::Lx}, Config::neighborCutoff};
    real_t volume = {subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2]};
    data::Atoms atoms = {fillDomainWithAtomsSC(subdomain, idx_c(Config::rho * volume), 1_r)};
    real_t rho = {real_c(atoms.numLocalAtoms) / volume};
    communication::GhostLayer ghostLayer;
    action::LennardJones LJ = {Config::rc, Config::sigma, Config::epsilon, 0.7_r * Config::sigma};

    HalfVerletList verletList;

public:
    NPT() = default;
};

TEST_P(NPT, pressure)
{
    auto targetTemperature = GetParam().targetTemperature;
    auto targetPressure = GetParam().targetPressure;

    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    util::ExponentialMovingAverage p(Config::weightingFactor);
    util::ExponentialMovingAverage T(Config::weightingFactor);
    T << analysis::getMeanKineticEnergy(atoms) * 2_r / 3_r;
    for (auto step = 0; step < Config::nsteps; ++step)
    {
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, Config::dt);

        if ((step > 200) && (step % 100 == 0))
        {
            action::BerendsenBarostat::apply(
                atoms, p, targetPressure, Config::gamma * 0.1_r, subdomain);
            volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
            maxAtomDisplacement = std::numeric_limits<real_t>::max();
        }
        action::BerendsenThermostat::apply(atoms, T, targetTemperature, Config::gamma);

        if (maxAtomDisplacement >= Config::skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            real_t gridDelta[3] = {
                Config::neighborCutoff, Config::neighborCutoff, Config::neighborCutoff};
            LinkedCellList linkedCellList(atoms.getPos(),
                                          0,
                                          atoms.numLocalAtoms,
                                          gridDelta,
                                          subdomain.minCorner.data(),
                                          subdomain.maxCorner.data());
            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(atoms, subdomain);
            verletList.build(atoms.getPos(),
                             0,
                             atoms.numLocalAtoms,
                             Config::neighborCutoff,
                             Config::cell_ratio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data(),
                             Config::estimatedMaxNeighbors);
        }
        else
        {
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        LJ.apply(atoms, verletList);

        if (step < 201)
        {
            p = util::ExponentialMovingAverage(Config::weightingFactor);
            T = util::ExponentialMovingAverage(Config::weightingFactor);
        }
        auto Ek = analysis::getKineticEnergy(atoms);
        p << 2_r * (Ek - LJ.getVirial()) / (3_r * volume);
        Ek /= real_c(atoms.numLocalAtoms);
        T << (2_r / 3_r) * Ek;

        ghostLayer.contributeBackGhostToReal(atoms);
        action::VelocityVerlet::postForceIntegrate(atoms, Config::dt);

        if (Config::bOutput && (step % Config::outputInterval == 0))
        {
            //            std::cout << step << " " << T << " " << p << " " << volume << std::endl;
        }
    }

    EXPECT_NEAR(T, targetTemperature, 0.1_r);
    EXPECT_NEAR(p, targetPressure, 0.2_r);
}

INSTANTIATE_TEST_SUITE_P(Pressure,
                         NPT,
                         ::testing::Values(Input{2.8_r, 9.1_r},
                                           Input{2.5_r, 8.5_r},
                                           Input{2.0_r, 8.0_r}));

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    return RUN_ALL_TESTS();
}
