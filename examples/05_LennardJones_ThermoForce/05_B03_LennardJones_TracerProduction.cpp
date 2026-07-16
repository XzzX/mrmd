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
#include <algorithm>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/LennardJones.hpp"
#include "action/LimitAcceleration.hpp"
#include "action/LimitVelocity.hpp"
#include "action/ThermodynamicForce.hpp"
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
#include "io/DumpProfile.hpp"
#include "io/DumpThermoForce.hpp"
#include "io/RestoreH5MD.hpp"
#include "io/RestoreThermoForce.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/IsInSymmetricSlab.hpp"
#include "util/PrintTable.hpp"
#include "util/simulationSetup.hpp"

using namespace mrmd;

/**
 * Configuration for the Argon NVE example simulation.
 */
struct Config
{
    // simulation time parameters
    idx_t nsteps = 40000001;  ///< number of steps to simulate
    real_t dt = 0.002;        ///< time step size in reduced units

    // input file parameters
    std::string fileRestoreH5MD = "thermodynamicForce_final.h5md";
    std::string fileRestoreTF;

    // interaction parameters
    static constexpr real_t sigma =
        1_r;  ///< distance at which LJ potential is zero in reduced units
    static constexpr real_t epsilon = 1_r;  ///< energy well depth of LJ potential in reduced units
    static constexpr real_t mass = 1_r;     ///< mass of one atom in reduced units
    static constexpr real_t maxVelocity =
        1_r;  ///< maximum initial velocity component in reduced units
    static constexpr real_t r_cut = 2.5_r * sigma;  ///< cutoff radius for LJ potential
    real_t r_cap_inner = 0.82417464_r * sigma;      ///< capping radius for LJ potential

    // neighbor list parameters
    static constexpr real_t skin = 0.3_r * sigma;           ///< skin thickness for neighbor list
    static constexpr real_t neighborCutoff = r_cut + skin;  ///< cutoff radius for neighbor list
    static constexpr real_t cell_ratio =
        1_r;  ///< ratio of cell size on Cartesian grid to cutoff radius for neighbor list
    static constexpr idx_t estimatedMaxNeighbors =
        60;  ///< estimated maximum number of neighbors per atom

    // thermostat parameters
    real_t target_temperature =
        1.5_r;  ///< target temperature during equilibration for thermostat in reduced units
    real_t gamma = 0.04_r / dt;  ///< friction coefficient for Langevin thermostat

    // application regions
    real_t innerIntRegionMin = 0_r;
    real_t innerIntRegionMax = 10_r * sigma + r_cut;
    real_t thermostatRegionMin = innerIntRegionMax;
    real_t thermostatRegionMax = 15_r * sigma;
    real_t thermoForceRegionMin = innerIntRegionMax;
    real_t thermoForceRegionMax = 14.5_r * sigma;

    // output parameters
    bool bOutput = true;                  ///< whether to output data files
    idx_t outputInterval = -1;            ///< interval for data file output (-1: no output)
    const std::string resName = "Argon";  ///< residue name for output files
    const std::vector<std::string> typeNames = {"Ar"};  ///< atom type names for output files

    std::string fileOut = "tracerProduction";  ///< base name for output files
    std::string fileOutH5MD = format("{0}.h5md", fileOut);
    std::string fileOutFinalGro = format("{0}_final.gro", fileOut);
    std::string fileOutFinalH5MD = format("{0}_final.h5md", fileOut);
};

void runTracerProduction(Config& config)
{
    // initialize
    data::Subdomain initialSubdomain;
    auto atoms = data::Atoms(0);

    // load data from file
    auto io = io::RestoreH5MD();
    io.restore(config.fileRestoreH5MD, initialSubdomain, atoms);

    // reinitialize subdomain with no ghost layer in x-direction
    data::Subdomain subdomain(
        {
            initialSubdomain.minCorner[0],
            initialSubdomain.minCorner[1],
            initialSubdomain.minCorner[2],
        },
        {
            initialSubdomain.maxCorner[0],
            initialSubdomain.maxCorner[1],
            initialSubdomain.maxCorner[2],
        },
        {0_r, initialSubdomain.ghostLayerThickness[1], initialSubdomain.ghostLayerThickness[1]});

    // calculate volume of the simulation domain
    const auto volume = subdomain.getVolume();

    // calculate and print initial density
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    // restore thermodynamic force from file
    std::cout << "restoring thermodynamic force from file" << std::endl;

    auto thermodynamicForce =
        io::restoreThermoForce(config.fileRestoreTF, subdomain, {rho}, {0_r}, true, false, 1);

    // set up ghost layer for periodic boundary conditions
    communication::GhostLayer ghostLayer;

    // set up neighbor list
    HalfVerletList verletList;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;

    // set up interaction potential and force calculation and application
    action::LennardJones lennardJonesInner(
        config.r_cut, config.sigma, config.epsilon, config.r_cap_inner);

    // calculate and print box center coordinates
    const auto boxCenter = subdomain.getCenter();

    std::cout << "x center: " << boxCenter[0] << std::endl;
    std::cout << "y center: " << boxCenter[1] << std::endl;
    std::cout << "z center: " << boxCenter[2] << std::endl;

    // set up different regions
    util::IsInSymmetricSlab isInInnerIntRegion({boxCenter[0], boxCenter[1], boxCenter[2]},
                                               config.innerIntRegionMin,
                                               config.innerIntRegionMax);
    util::IsInSymmetricSlab isInThermostatRegion({boxCenter[0], boxCenter[1], boxCenter[2]},
                                                 config.thermostatRegionMin,
                                                 config.thermostatRegionMax);
    util::IsInSymmetricSlab isInThermoForceRegion({boxCenter[0], boxCenter[1], boxCenter[2]},
                                                  config.thermoForceRegionMin,
                                                  config.thermoForceRegionMax);

    // set up thermostat for temperature control
    action::VelocityVerletLangevinThermostat langevinIntegrator(config.gamma,
                                                                config.target_temperature);

    // set up timer for runtime measurement
    Kokkos::Timer timer;

    // set up mean square displacement analysis
    analysis::MeanSquareDisplacement meanSquareDisplacement;
    meanSquareDisplacement.reset(atoms);
    auto msd = 0_r;

    // set up variables for counting particle flux across the domain
    analysis::CountingPlane countingPlane(
        {boxCenter[0] + (config.innerIntRegionMax - config.r_cut), boxCenter[1], boxCenter[2]},
        {1_r, 0_r, 0_r});
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
        maxAtomDisplacement +=
            langevinIntegrator.preForceIntegrate_apply_if(atoms, config.dt, isInThermostatRegion);

        // stop counting particle flux across the plane and calculate flux
        flux += countingPlane.stopCounting(atoms);

        // check if neighbor list needs to be rebuilt
        if (maxAtomDisplacement >=
            config.skin *
                0.5_r)  // the condition is on half the skin thickness because in principle two
                        // atoms may both move half the skin thickness towards each other
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            // reinsert atoms that left the domain according to periodic boundary conditions
            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            // create ghost atoms in the ghost layer beyond the periodic boundaries
            ghostLayer.createGhostAtoms(atoms, subdomain);

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
        else
        {
            // update ghost atom positions in the ghost layer according to periodic boundary
            // conditions
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        // reset forces to zero
        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        // compute and apply forces
        thermodynamicForce.applyInterpolated_if(atoms, isInThermoForceRegion);

        lennardJonesInner.apply_if(
            atoms,
            verletList,
            KOKKOS_LAMBDA(const real_t x1,
                          const real_t y1,
                          const real_t z1,
                          const real_t x2,
                          const real_t y2,
                          const real_t z2) {
                return (isInInnerIntRegion(x1, y1, z1) && isInInnerIntRegion(x2, y2, z2));
            });

        // contribute forces calculated on ghost atoms back to real atoms
        ghostLayer.contributeBackGhostToReal(atoms);

        // integrate equations of motion after force calculation
        langevinIntegrator.postForceIntegrate(atoms, config.dt);

        // handle output and statistics
        if (config.bOutput && (step % config.outputInterval == 0))
        {
            // calculate statistics
            auto E0 = (lennardJonesInner.getEnergy()) / real_c(atoms.numLocalAtoms);
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
                  << E0 + Ek << " " << p << " " << msd << " " << flux << " "

                  << atoms.numLocalAtoms << " " << atoms.numGhostAtoms << " " << std::endl;

            // reset flux counter
            flux = 0;

            // phase point output
            dumpH5MD.dumpStep(subdomain, atoms, step, config.dt);
        }
    }
    if (config.bOutput)
    {
        dumpH5MD.close();

        // final phase point output
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

int main(int argc, char* argv[])  // NOLINT
{
    // initialize
    Kokkos::initialize(argc, argv);

    // print Kokkos execution space
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    // initialize simulation configuration with command line interface
    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-d,--tstep", config.dt, "time step");
    app.add_option("-o,--outint", config.outputInterval, "output interval");
    app.add_option("-i,--inpfile", config.fileRestoreH5MD, "input file name");
    app.add_option("-f,--outfile", config.fileOut, "output file name");

    app.add_option("--temp", config.target_temperature, "target temperature");
    app.add_option("--friction", config.gamma, "friction coefficient for langevin thermostat");

    app.add_option("--forceinp", config.fileRestoreTF, "input file for the thermodynamics force");
    app.add_option(
        "--rcapinner", config.r_cap_inner, "capping radius for inner Lennard-Jones potential");

    app.add_option(
        "--innermin", config.innerIntRegionMin, "inner interacting region minimum coordinate");
    app.add_option(
        "--innermax", config.innerIntRegionMax, "inner interacting region maximum coordinate");
    app.add_option(
        "--thermostatmin", config.thermostatRegionMin, "thermostat region minimum coordinate");
    app.add_option(
        "--thermostatmax", config.thermostatRegionMax, "thermostat region maximum coordinate");
    app.add_option("--thermoforcemin",
                   config.thermoForceRegionMin,
                   "thermodynamic force region minimum coordinate");
    app.add_option("--thermoforcemax",
                   config.thermoForceRegionMax,
                   "thermodynamic force region maximum coordinate");

    CLI11_PARSE(app, argc, argv);

    config.fileOutH5MD = format("{0}.h5md", config.fileOut);
    config.fileOutFinalGro = format("{0}_final.gro", config.fileOut);
    config.fileOutFinalH5MD = format("{0}_final.h5md", config.fileOut);

    if (config.outputInterval < 0) config.bOutput = false;

    // set up run simulation
    runTracerProduction(config);

    // finalize
    Kokkos::finalize();

    return EXIT_SUCCESS;
}