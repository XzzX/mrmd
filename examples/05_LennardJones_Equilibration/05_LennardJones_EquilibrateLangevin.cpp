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

#include "action/LennardJones.hpp"
#include "action/VelocityVerletLangevinThermostat.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpGRO.hpp"
#include "io/DumpH5MD.hpp"
#include "io/RestoreH5MD.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/ExponentialMovingAverage.hpp"
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    // time parameters
    idx_t nsteps = 100001;  ///< number of steps to simulate
    real_t dt = 0.002_r;    ///< time step size in reduced units

    // input file parameters
    std::string fileRestoreH5MD =
        "equilibrateBerendsen_final.h5md";  ///< name of the file to restore the initial phase point
                                            ///< from

    // interaction parameters
    static constexpr real_t sigma =
        1_r;  ///< distance at which LJ potential is zero in reduced units
    static constexpr real_t epsilon = 1_r;  ///< energy well depth of LJ potential in reduced units
    static constexpr real_t mass = 1_r;     ///< mass of one atom in reduced units
    static constexpr real_t r_cut = 2.5_r * sigma;  ///< cutoff radius for LJ potential
    static constexpr real_t r_cap = 0_r;            ///< capping radius for LJ potential
    static constexpr bool doShift =
        true;  ///< whether to shift the LJ potential to zero at the cutoff radius

    // pressure parameters
    real_t pressure_averaging_coefficient =
        0.02_r;  ///< coefficient for exponential moving average of pressure

    // thermostatting parameters
    real_t temperature = 1.5_r;     ///< target temperature for thermostat in reduced units
    real_t friction = 0.04_r / dt;  ///< friction coefficient for Langevin thermostat
    real_t temperature_averaging_coefficient =
        0.2_r;  ///< coefficient for exponential moving average of temperature

    // neighbor-list parameters
    static constexpr real_t skin = 0.3_r * sigma;           ///< skin thickness for neighbor list
    static constexpr real_t neighborCutoff = r_cut + skin;  ///< cutoff radius for neighbor list
    static constexpr real_t cell_ratio =
        1_r;  ///< ratio of cell size on Cartesian grid to cutoff radius for neighbor list
    static constexpr idx_t estimatedMaxNeighbors =
        60;  ///< estimated maximum number of neighbors per atom

    // output parameters
    bool bOutput = true;                  ///< whether to output data files
    idx_t outputInterval = -1;            ///< interval for data file output (-1: no output)
    const std::string resName = "Argon";  ///< residue name for output files
    const std::vector<std::string> typeNames = {"Ar"};  ///< atom type names for output files

    std::string fileOut = "equilibrateLangevin";  ///< base name for output files
    std::string fileOutH5MD;
    std::string fileOutTF;
    std::string fileOutFinalGRO;
    std::string fileOutFinalH5MD;
};

void equilibrateLangevin(Config& config)
{
    // initialize
    data::Subdomain subdomain;
    auto atoms = data::Atoms(0);

    // load data from file
    auto io = io::RestoreH5MD();
    io.restore(config.fileRestoreH5MD, subdomain, atoms);

    const auto volume = subdomain.getVolume();
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    // output management
    auto dump = io::DumpH5MD("J-Hizzle");

    // technical setup
    communication::GhostLayer ghostLayer;
    action::LennardJones LJ(config.r_cut, config.sigma, config.epsilon, 0.5_r * config.sigma);
    HalfVerletList verletList;
    action::VelocityVerletLangevinThermostat langevinIntegrator(config.friction,
                                                                config.temperature);
    Kokkos::Timer timer;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    util::ExponentialMovingAverage currentPressure(config.pressure_averaging_coefficient);
    util::ExponentialMovingAverage currentTemperature(config.temperature_averaging_coefficient);

    // output management
    auto dumpH5MD = io::DumpH5MD("J-Hizzle");
    if (config.bOutput)
    {
        util::printTable(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
        util::printTableSep(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
        dumpH5MD.open(config.fileOutH5MD, subdomain, atoms);
    }

    // main integration loop
    for (auto step = 0; step < config.nsteps; ++step)
    {
        maxAtomDisplacement += langevinIntegrator.preForceIntegrate(atoms, config.dt);

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
        langevinIntegrator.postForceIntegrate(atoms, config.dt);

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

            // phase point output
            dumpH5MD.dumpStep(subdomain, atoms, step, config.dt);
        }
    }

    if (config.bOutput)
    {
        dumpH5MD.close();

        // final phase point output
        dumpH5MD.dump(config.fileOutFinalH5MD, subdomain, atoms);

        io::dumpGRO(config.fileOutFinalGRO,
                    atoms,
                    subdomain,
                    0,
                    config.resName,
                    config.resName,
                    config.typeNames,
                    false,
                    true);
    }

    // write performance data to file
    auto time = timer.seconds();
    std::cout << time << std::endl;
    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");
    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::initialize(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"follow-up NVT equilibration run of Lennard-Jones fluid with Langevin thermostat"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-o,--outint", config.outputInterval, "output interval");
    app.add_option("-i,--inpfile", config.fileRestoreH5MD, "input file name");
    app.add_option("-T,--temperature", config.temperature, "thermostat target temperature");
    app.add_option("--friction", config.friction, "friction coefficient for thermostat");
    app.add_option("-f,--outfile", config.fileOut, "output file name");

    CLI11_PARSE(app, argc, argv);

    config.fileOutH5MD = format("{0}.h5md", config.fileOut);
    config.fileOutTF = format("{0}_tf.txt", config.fileOut);
    config.fileOutFinalH5MD = format("{0}_final.h5md", config.fileOut);
    config.fileOutFinalGRO = format("{0}_final.gro", config.fileOut);

    if (config.outputInterval < 0) config.bOutput = false;
    equilibrateLangevin(config);

    Kokkos::finalize();

    return EXIT_SUCCESS;
}