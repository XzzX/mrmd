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
#include "analysis/CountingPlane.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/MeanSquareDisplacement.hpp"
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
#include "util/IsInSymmetricSlab.hpp"
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    // time parameters
    idx_t nsteps = 200000001;  ///< number of steps to simulate
    real_t dt = 0.002_r;       ///< time step size in reduced units

    // input file parameters
    std::string fileRestoreH5MD =
        "equilibrateLangevin_final.h5md";  ///< name of the file to restore the initial phase point
                                           ///< from

    // interaction parameters
    static constexpr real_t sigma =
        1_r;  ///< distance at which LJ potential is zero in reduced units
    static constexpr real_t epsilon = 1_r;  ///< energy well depth of LJ potential in reduced units
    static constexpr real_t mass = 1_r;     ///< mass of one atom in reduced units
    static constexpr real_t r_cut = 2.5_r * sigma;  ///< cutoff radius for LJ potential
    real_t r_cap = 0_r;                             ///< capping radius for LJ potential

    // neighbor list parameters
    static constexpr real_t skin = 0.3_r * sigma;           ///< skin thickness for neighbor list
    static constexpr real_t neighborCutoff = r_cut + skin;  ///< cutoff radius for neighbor list
    static constexpr real_t cell_ratio =
        1_r;  ///< ratio of cell size on Cartesian grid to cutoff radius for neighbor list
    static constexpr idx_t estimatedMaxNeighbors =
        60;  ///< estimated maximum number of neighbors per atom

    // thermostat parameters
    real_t temperature =
        1.5_r;  ///< target temperature during equilibration for thermostat in reduced units
    real_t friction = 0.04_r / dt;  ///< friction coefficient for Langevin thermostat

    // output parameters
    bool bOutput = true;                  ///< whether to output data files
    idx_t outputInterval = -1;            ///< interval for data file output (-1: no output)
    const std::string resName = "Argon";  ///< residue name for output files
    const std::vector<std::string> typeNames = {"Ar"};  ///< atom type names for output files

    std::string fileOut = "productionAtomistic";  ///< base name for output files
    std::string fileOutH5MD = format("{0}.h5md", fileOut);
    std::string fileOutFinalGro = format("{0}_final.gro", fileOut);
    std::string fileOutFinalH5MD = format("{0}_final.h5md", fileOut);
};

void productionAtomistic(Config& config)
{
    // initialize
    data::Subdomain subdomain;
    auto atoms = data::Atoms(0);

    // load data from file
    auto io = io::RestoreH5MD();
    io.restore(config.fileRestoreH5MD, subdomain, atoms);

    // calculate volume of the simulation domain
    const auto volume = subdomain.getVolume();

    // calculate and print initial density
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    // set up ghost layer for periodic boundary conditions
    communication::GhostLayer ghostLayer;

    // set up neighbor list
    HalfVerletList verletList;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;

    // set up interaction potential and force calculation and application
    action::LennardJones lennardJones(config.r_cut, config.sigma, config.epsilon, config.r_cap);

    // calculate and print box center coordinates
    const auto boxCenter = subdomain.getCenter();

    std::cout << "x center: " << boxCenter[0] << std::endl;
    std::cout << "y center: " << boxCenter[1] << std::endl;
    std::cout << "z center: " << boxCenter[2] << std::endl;

    // set up thermostat for temperature control
    action::VelocityVerletLangevinThermostat langevinIntegrator(config.friction,
                                                                config.temperature);

    // set up timer for runtime measurement
    Kokkos::Timer timer;

    // set up mean square displacement analysis
    analysis::MeanSquareDisplacement meanSquareDisplacement;
    meanSquareDisplacement.reset(atoms);
    auto msd = 0_r;

    // set up variables for counting particle flux across the domain
    analysis::CountingPlane countingPlane(
        {boxCenter[0] + 10_r * config.sigma, boxCenter[1], boxCenter[2]}, {1_r, 0_r, 0_r});
    int64_t flux = 0;

    // output management
    auto dumpH5MD = io::DumpH5MD("J-Hizzle");
    std::ofstream fStat("statistics.txt");
    if (config.bOutput)
    {
        // print table header for simulation statistics
        util::printTable(
            "step", "time", "T", "Ek", "E0", "E", "p", "msd", "flux", "Nlocal", "Nghost");
        util::printTableSep(
            "step", "time", "T", "Ek", "E0", "E", "p", "msd", "flux", "Nlocal", "Nghost");

        // phase point output setup
        dumpH5MD.open(config.fileOutH5MD, subdomain, atoms);
    }

    // main simulation loop
    for (auto step = 0; step < config.nsteps; ++step)
    {
        // start counting particle flux across the plane
        countingPlane.startCounting(atoms);

        // integrate equations of motion with local Langevin thermostat during production phase
        maxAtomDisplacement += langevinIntegrator.preForceIntegrate(atoms, config.dt);

        // stop counting particle flux across the plane and calculate flux
        flux += countingPlane.stopCounting(atoms);

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

        // reset forces to zero
        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        // compute and apply forces
        lennardJones.apply(atoms, verletList);

        // contribute forces calculated on ghost atoms back to real atoms
        ghostLayer.contributeBackGhostToReal(atoms);

        // integrate equations of motion after force calculation
        langevinIntegrator.postForceIntegrate(atoms, config.dt);

        // handle output and statistics
        if (config.bOutput && (step % config.outputInterval == 0))
        {
            // calculate statistics
            auto E0 = (lennardJones.getEnergy()) / real_c(atoms.numLocalAtoms);
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (2_r / 3_r) * Ek;
            auto p = analysis::getPressure(atoms, subdomain);
            msd = meanSquareDisplacement.calc(atoms, subdomain) /
                  (real_c(config.outputInterval) * config.dt);
            meanSquareDisplacement.reset(atoms);

            // print statistics to console
            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             p,
                             msd,
                             flux,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            // dump statistics to file
            fStat << step << " " << timer.seconds() << " " << T << " " << Ek << " " << E0 << " "
                  << E0 + Ek << " " << p << " " << msd << " " << flux << " " << atoms.numLocalAtoms
                  << " " << atoms.numGhostAtoms << " " << std::endl;

            // reset flux counter
            flux = 0;

            // phase point output
            dumpH5MD.dumpStep(subdomain, atoms, step, config.dt);
        }
    }

    if (config.bOutput)
    {
        dumpH5MD.close();

        // final microstates output
        dumpH5MD.dump(config.fileOutFinalH5MD, subdomain, atoms);

        // close statistics file
        fStat.close();
        auto time = timer.seconds();
        std::cout << time << std::endl;

        io::dumpGRO(config.fileOutFinalGro,
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
    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");
    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-d,--tstep", config.dt, "time step");
    app.add_option("-o,--outint", config.outputInterval, "output interval");
    app.add_option("-i,--inpfile", config.fileRestoreH5MD, "input file name");
    app.add_option("-f,--outfile", config.fileOut, "output file name");

    app.add_option("-T,--temperature", config.temperature, "thermostat target temperature");
    app.add_option("--friction", config.friction, "friction coefficient for thermostat");

    app.add_option("--rcap", config.r_cap, "capping radius for Lennard-Jones potential");

    CLI11_PARSE(app, argc, argv);

    config.fileOutH5MD = format("{0}.h5md", config.fileOut);
    config.fileOutFinalGro = format("{0}_final.gro", config.fileOut);
    config.fileOutFinalH5MD = format("{0}_final.h5md", config.fileOut);

    if (config.outputInterval < 0) config.bOutput = false;

    productionAtomistic(config);

    Kokkos::finalize();

    return EXIT_SUCCESS;
}