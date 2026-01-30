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

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/BerendsenThermostat.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/LimitAcceleration.hpp"
#include "action/LimitVelocity.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/MeanSquareDisplacement.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "initialization.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/PrintTable.hpp"
#include "util/simulationSetup.hpp"

using namespace mrmd;

/**
 * Configuration for the Argon NVE example simulation.
 */
struct Config
{
    // simulation time parameters
    idx_t nsteps = 400001;               ///< number of steps to simulate
    static constexpr real_t dt = 0.002;  ///< time step size in reduced units

    // interaction parameters
    static constexpr real_t sigma =
        1_r;  ///< distance at which LJ potential is zero in reduced units
    static constexpr real_t epsilon = 1_r;  ///< energy well depth of LJ potential in reduced units
    static constexpr real_t mass = 1_r;     ///< mass of one atom in reduced units
    static constexpr real_t maxVelocity =
        1_r;  ///< maximum initial velocity component in reduced units
    static constexpr real_t r_cut = 2.5_r * sigma;  ///< cutoff radius for LJ potential
    static constexpr real_t r_cap = 0.7_r * sigma;  ///< capping radius for LJ potential

    // neighbor list parameters
    static constexpr real_t skin = 0.1_r * sigma;           ///< skin thickness for neighbor list
    static constexpr real_t neighborCutoff = r_cut + skin;  ///< cutoff radius for neighbor list
    static constexpr real_t cell_ratio =
        1_r;  ///< ratio of cell size on Cartesian grid to cutoff radius for neighbor list
    static constexpr idx_t estimatedMaxNeighbors =
        60;  ///< estimated maximum number of neighbors per atom

    // system parameters
    static constexpr idx_t numAtoms = 16 * 16 * 16;  ///< number of atoms in the simulation
    real_t Lx = 20_r * sigma;                        ///< box edge length

    // equilibration parameters
    idx_t nstepsEq = 100000;  ///< number of equilibration steps
    real_t temperature =
        1.5_r;  ///< target temperature during equilibration for thermostat in reduced units
    static constexpr real_t gamma = 0.04_r / dt;  ///< friction coefficient for Langevin thermostat

    // output parameters
    bool bOutput = true;                  ///< whether to output data files
    idx_t outputInterval = -1;            ///< interval for data file output (-1: no output)
    const std::string resName = "Argon";  ///< residue name for output files
    const std::vector<std::string> typeNames = {"Ar"};  ///< atom type names for output files
};

void runSimulation(Config& config)
{
    // initialize simulation domain
    auto subdomain =
        data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Lx, config.Lx}, config.neighborCutoff);

    // calculate volume of the simulation domain
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];

    // initialize atoms randomly in the domain
    auto atoms =
        util::fillDomainWithAtoms(subdomain, config.numAtoms, config.maxVelocity, config.mass);

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
    action::LennardJones LJ(config.r_cut, config.sigma, config.epsilon, config.r_cap);

    // set up thermostat for temperature control during equilibration
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);

    // set up timer for runtime measurement
    Kokkos::Timer timer;

    // set up mean square displacement analysis
    analysis::MeanSquareDisplacement meanSquareDisplacement;
    meanSquareDisplacement.reset(atoms);
    auto msd = 0_r;

    // print table header for simulation statistics
    util::printTable("step", "time", "T", "Ek", "E0", "E", "p", "msd", "Nlocal", "Nghost");
    util::printTableSep("step", "time", "T", "Ek", "E0", "E", "p", "msd", "Nlocal", "Nghost");

    // open statistics file for writing simulation statistics
    std::ofstream fStat("statistics.txt");

    // main simulation loop
    for (auto step = 0; step < config.nsteps; ++step)
    {
        // integrate equations of motion - first half step (drift)
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        // reinsert atoms that left the domain according to periodic boundary conditions
        ghostLayer.exchangeRealAtoms(atoms, subdomain);

        // create ghost atoms in the ghost layer beyond the periodic boundaries
        ghostLayer.createGhostAtoms(atoms, subdomain);

        // check if neighbor list needs to be rebuilt
        if (maxAtomDisplacement >=
            config.skin *
                0.5_r)  // the condition is on half the skin thickness because in principle two
                        // atoms may both move half the skin thickness towards each other
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            // rebuild neighbor list
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

        // reset forces to zero
        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        // compute and apply forces
        LJ.apply(atoms, verletList);

        // contribute forces calculated on ghost atoms back to real atoms
        ghostLayer.contributeBackGhostToReal(atoms);

        // handle output and statistics
        if (config.bOutput && (step % config.outputInterval == 0))
        {
            // calculate statistics
            auto E0 = LJ.getEnergy() / real_c(atoms.numLocalAtoms);
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (2_r / 3_r) * Ek;
            auto p = analysis::getPressure(atoms, subdomain);
            msd =
                meanSquareDisplacement.calc(atoms, subdomain) / (config.outputInterval * config.dt);
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
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            // dump statistics to file
            fStat << step << " " << timer.seconds() << " " << T << " " << Ek << " " << E0 << " "
                  << E0 + Ek << " " << p << " " << msd << " " << atoms.numLocalAtoms << " "
                  << atoms.numGhostAtoms << " " << std::endl;
        }

        // check if still during equilibration phase
        if (step <= config.nstepsEq)
        {
            // apply Langevin thermostat for temperature control during equilibration
            langevinThermostat.apply(atoms);
        }

        // integrate equations of motion - second half step (kick)
        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);
    }

    // close statistics file
    fStat.close();
    auto time = timer.seconds();
    std::cout << time << std::endl;

    // write performance data to file
    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");
    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalAtoms << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])  // NOLINT
{
    // initialize Kokkos and MPI environment
    initialize(argc, argv);

    // print Kokkos execution space
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    // initialize simulation configuration with command line interface
    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "total number of simulation steps");
    app.add_option("-L,--length", config.Lx, "simulation box diameter");
    app.add_option("-e,--numeq", config.nstepsEq, "number of equilibration steps");
    app.add_option(
        "-T,--temperature",
        config.temperature,
        "temperature of the Langevin thermostat (negative numbers deactivate the thermostat)");
    app.add_option("-o,--output", config.outputInterval, "output interval");
    CLI11_PARSE(app, argc, argv);

    // reset output parameter if output interval is negative
    if (config.outputInterval < 0) config.bOutput = false;

    // set up, equilibrate in NVT and run production in NVE
    runSimulation(config);

    // finalize Kokkos and MPI environment
    finalize();

    return EXIT_SUCCESS;
}