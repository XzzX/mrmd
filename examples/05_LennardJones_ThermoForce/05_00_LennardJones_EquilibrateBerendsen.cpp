// Copyright 2024 Sebastian Eibl
// Copyright 2026 Julian Friedrich Hille
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

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <format>

#include "action/BerendsenThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpGRO.hpp"
#include "io/DumpH5MD.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/ExponentialMovingAverage.hpp"
#include "util/PrintTable.hpp"
#include "util/simulationSetup.hpp"

using namespace mrmd;

struct Config
{
    // output parameters
    bool bOutput = true;
    std::string fileOut = "equilibrateBerendsen";
    std::string fileOutH5MD = format("{0}.h5md", fileOut);
    std::string fileOutGro = format("{0}.gro", fileOut);
    idx_t outputInterval = 1000;

    // time parameters
    idx_t nsteps = 100001;
    real_t dt = 0.001;

    // system parameters
    const std::string resName = "Argon";
    const std::vector<std::string> typeNames = {"Ar"};

    // interaction parameters
    real_t sigma = 1_r;
    real_t epsilon = 1_r;
    real_t rCut = 2.5_r;
    real_t rCap = 0.0_r;
    bool doShift = true;

    // pressure parameters
    real_t pressure_averaging_coefficient = 0.02;

    // thermostatting parameters
    real_t temperature = 1.5_r;
    real_t friction = 1.0_r;
    real_t temperature_averaging_coefficient = 0.2_r;
    idx_t thermostat_interval = 1;

    // box parameters
    real_t Lx = 45.0_r;
    real_t Ly = 30.0_r;
    real_t Lz = 30.0_r;
    idx_t numAtoms = 15000;

    // neighbor-list parameters
    real_t cell_ratio = 1.0_r;
    real_t skin = 0.3;
    real_t neighborCutoff = rCut + skin;
    idx_t estimatedMaxNeighbors = 60;
};

void equilibrateBerendsen(Config& config)
{
    // initialize
    auto subdomain =
        data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Ly, config.Lz}, config.neighborCutoff);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto atoms = util::fillDomainWithAtoms(subdomain, config.numAtoms, 1_r, 1_r);
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    // output management
    auto dump = io::DumpH5MD("J-Hizzle");
    io::dumpGRO("equilibrateBerendsenInitial.gro",
                atoms,
                subdomain,
                0_r,
                "Argon",
                config.resName,
                config.typeNames,
                false,
                true);

    // technical setup
    communication::GhostLayer ghostLayer;
    action::LennardJones LJ(config.rCut, config.sigma, config.epsilon, 0.5_r * config.sigma);
    HalfVerletList verletList;
    Kokkos::Timer timer;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    util::ExponentialMovingAverage currentPressure(config.pressure_averaging_coefficient);
    util::ExponentialMovingAverage currentTemperature(config.temperature_averaging_coefficient);
    currentTemperature << analysis::getMeanKineticEnergy(atoms) * 2_r / 3_r;

    // thermodynamic observables table
    if (config.bOutput)
    {
        util::printTable(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
        util::printTableSep(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
    }

    for (auto step = 0; step < config.nsteps; ++step)
    {
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        if (step % config.thermostat_interval == 0)
        {
            action::BerendsenThermostat::apply(atoms,
                                               currentTemperature,
                                               config.temperature,
                                               config.friction);
        }

        if (maxAtomDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

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

        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        LJ.apply(atoms, verletList);

        auto Ek = analysis::getKineticEnergy(atoms);
        currentPressure << 2_r * (Ek - LJ.getVirial()) / (3_r * volume);
        Ek /= real_c(atoms.numLocalAtoms);
        currentTemperature << (2_r / 3_r) * Ek;

        ghostLayer.contributeBackGhostToReal(atoms);
        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            util::printTable(step,
                             timer.seconds(),
                             currentTemperature,
                             currentPressure,
                             volume,
                             Ek,
                             LJ.getEnergy() / real_c(atoms.numLocalAtoms),
                             Ek + LJ.getEnergy() / real_c(atoms.numLocalAtoms),
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);
        }
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;

    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();

    dump.dump(config.fileOutH5MD, subdomain, atoms);
    io::dumpGRO(config.fileOutGro,
                atoms,
                subdomain,
                0,
                config.resName,
                config.resName,
                config.typeNames,
                false,
                true);
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-o,--output", config.outputInterval, "output interval");
    app.add_option("--xlength", config.Lx, "x length of the box");
    app.add_option("--ylength", config.Ly, "y length of the box");
    app.add_option("--zlength", config.Lz, "z length of the box");
    app.add_option("--numAtoms", config.numAtoms, "number of atoms");
    app.add_option("--temp", config.temperature, "target temperature");
    app.add_option("-f,--outfile", config.fileOut, "output file name");

    CLI11_PARSE(app, argc, argv);

    config.fileOutH5MD = format("{0}.h5md", config.fileOut);
    config.fileOutGro = format("{0}.gro", config.fileOut);

    if (config.outputInterval < 0) config.bOutput = false;

    equilibrateBerendsen(config);

    Kokkos::finalize();

    return EXIT_SUCCESS;
}