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

#include <fmt/format.h>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/BerendsenThermostat.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/LimitAcceleration.hpp"
#include "action/LimitVelocity.hpp"
#include "action/ThermodynamicForce.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/AxialDensityProfile.hpp"
#include "analysis/Fluctuation.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/MeanSquareDisplacement.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"
#include "io/DumpGRO.hpp"
#include "io/RestoreTXT.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = true;
    idx_t outputInterval = -1;

    idx_t nsteps = 6000001;
    static constexpr idx_t numAtoms = 10 * 10 * 10;

    static constexpr real_t sigma = 0.34_r;        ///< units: nm
    static constexpr real_t epsilon = 0.993653_r;  ///< units: kJ/mol
    static constexpr real_t mass = 39.948_r;       ///< units: u

    static constexpr real_t rc = 2.3_r * sigma;
    static constexpr real_t skin = 0.1;
    static constexpr real_t neighborCutoff = rc + skin;

    static constexpr real_t dt = 0.00217;  ///< units: ps
    real_t temperature = 3.0_r;
    real_t gamma = 0.7_r / dt;

    real_t Lx = 8_r;  ///< units: nm
    real_t Ly = 8_r;  ///< units: nm
    real_t Lz = 8_r;  ///< units: nm

    real_t cell_ratio = 1.0_r;

    idx_t estimatedMaxNeighbors = 60;

    idx_t densityStart = 2000001;
    idx_t densityDeadTime = 10001;
    idx_t densitySamplingInterval = 10;
    idx_t densityUpdateInterval = 2000;
    real_t densityBinWidth = 0.5_r;
    real_t smoothingSigma = 2_r;
    real_t smoothingIntensity = 2_r;
    // thermodynamic force parameters
    real_t thermodynamicForceModulation = 2.0_r;
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
    auto type = atoms.getType();
    auto charge = atoms.getCharge();

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

        mass(idx) = Config::mass;
        type(idx) = 0;
        charge(idx) = 0_r;
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

void LJ(Config& config)
{
    auto subdomain = data::Subdomain({-config.Lx, -config.Ly, -config.Lz},
                                     {config.Lx, config.Ly, config.Lz},
                                     config.neighborCutoff);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numAtoms, 1_r);
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    io::dumpGRO("atoms_initial.gro", atoms, subdomain, 0_r, "Argon", false);

    communication::GhostLayer ghostLayer;
    action::LennardJones LJ(config.rc, config.sigma, config.epsilon, 0.7_r * config.sigma);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    action::ThermodynamicForce thermodynamicForce(
        rho, subdomain, config.densityBinWidth, config.thermodynamicForceModulation);
    analysis::MeanSquareDisplacement meanSquareDisplacement;
    meanSquareDisplacement.reset(atoms);
    auto msd = 0_r;
    HalfVerletList verletList;
    Kokkos::Timer timer;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    util::printTable("step", "time", "T", "Ek", "E0", "E", "p", "msd", "fluc", "Nlocal", "Nghost");
    util::printTableSep(
        "step", "time", "T", "Ek", "E0", "E", "p", "msd", "fluc", "Nlocal", "Nghost");

    std::ofstream fThermodynamicForceOut("thermodynamicForce.txt");
    std::ofstream fDensityProfile("densityProfile.txt");
    std::ofstream fStat("statistics.txt");
    for (auto step = 0; step < config.nsteps; ++step)
    {
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        if (maxAtomDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            //            real_t gridDelta[3] = {
            //                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            //            LinkedCellList linkedCellList(atoms.getPos(),
            //                                          0,
            //                                          atoms.numLocalAtoms,
            //                                          gridDelta,
            //                                          subdomain.minCorner.data(),
            //                                          subdomain.maxCorner.data());
            //            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(atoms, subdomain);
            verletList.build(atoms.getPos(),
                             0,
                             atoms.numLocalAtoms,
                             config.neighborCutoff,
                             config.cell_ratio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data(),
                             config.estimatedMaxNeighbors);
            ++rebuildCounter;
        }
        else
        {
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        auto pos = atoms.getPos();
        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        if (step > config.densityStart)
        {
            if (step % config.densitySamplingInterval == 0)
            {
                thermodynamicForce.sample(atoms);
            }

            if (step % config.densityUpdateInterval == 0)
            {
                thermodynamicForce.update(config.smoothingSigma, config.smoothingIntensity);
                config.densityStart = step + config.densityDeadTime;
            }
        }

        thermodynamicForce.apply(atoms);

        {
            auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
            auto kernel = KOKKOS_LAMBDA(const idx_t idx)
            {
                auto f = (pos(idx, 0) > 0) ? 10_r : -10_r;
                force(idx, 0) += f;
            };
            Kokkos::parallel_for("external-force", policy, kernel);
        }
        LJ.apply(atoms, verletList);

        ghostLayer.contributeBackGhostToReal(atoms);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto E0 = LJ.getEnergy() / real_c(atoms.numLocalAtoms);
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (2_r / 3_r) * Ek;
            //            std::cout << "system momentum: " << systemMomentum[0] << " | " <<
            //            systemMomentum[1]
            //                      << " | " << systemMomentum[2] << std::endl;
            //            std::cout << "rebuild counter: " << rebuildCounter << std::endl;
            auto p = analysis::getPressure(atoms, subdomain);
            auto densityProfile = analysis::getAxialDensityProfile(atoms.numLocalAtoms,
                                                                   atoms.getPos(),
                                                                   atoms.getType(),
                                                                   1,
                                                                   subdomain.minCorner[0],
                                                                   subdomain.maxCorner[0],
                                                                   10,
                                                                   COORD_X);
            densityProfile.scale(1_r / (densityProfile.binSize * config.Ly * config.Lz * 4));
            auto fluctuation = analysis::getFluctuation(densityProfile, rho, 0);

            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             p,
                             msd,
                             fluctuation,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            fStat << step << " " << timer.seconds() << " " << T << " " << Ek << " " << E0 << " "
                  << E0 + Ek << " " << p << " " << msd << " " << fluctuation << " "
                  << atoms.numLocalAtoms << " " << atoms.numGhostAtoms << " " << std::endl;

            auto Fth = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           thermodynamicForce.getForce(0));
            for (auto i = 0; i < Fth.extent(0); ++i)
            {
                fThermodynamicForceOut << Fth(i) << " ";
            }
            fThermodynamicForceOut << std::endl;

            auto h_densityProfile =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), densityProfile.data);
            for (auto i = 0; i < h_densityProfile.extent(0); ++i)
            {
                fDensityProfile << h_densityProfile(i, 0) << " ";
            }
            fDensityProfile << std::endl;

            //            io::dumpCSV(fmt::format("atoms_{:0>6}.csv", step), atoms, false);
        }

        if (step % 1000 == 0)
        {
            msd = meanSquareDisplacement.calc(atoms, subdomain) / (1000_r * config.dt);
            meanSquareDisplacement.reset(atoms);
        }

        langevinThermostat.apply(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;
    fStat.close();
    fThermodynamicForceOut.close();
    fDensityProfile.close();

    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-L,--length", config.Lx, "simulation box diameter");
    app.add_option(
        "-T,--temperature",
        config.temperature,
        "temperature of the Langevin thermostat (negative numbers deactivate the thermostat)");
    app.add_option("-o,--output", config.outputInterval, "output interval");
    CLI11_PARSE(app, argc, argv);

    if (config.outputInterval < 0) config.bOutput = false;
    LJ(config);

    return EXIT_SUCCESS;
}