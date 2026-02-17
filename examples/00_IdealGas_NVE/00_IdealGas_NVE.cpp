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

#include "action/VelocityVerlet.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"
#include "util/simulationSetup.hpp"

using namespace mrmd;

/**
 * Configuration for the ideal gas example simulation in NVE ensemble.
 */
struct Config
{
    idx_t nsteps = 2001;  ///< number of steps to simulate
    real_t dt = 0.005;    ///< time step size in reduced units
};

void runIdealGas(const Config& config)
{
    // initialize simulation domain
    auto subdomain = data::Subdomain({0_r, 0_r, 0_r}, {100_r, 100_r, 100_r}, 1_r);

    // initialize atoms randomly in the domain
    auto atoms = util::fillDomainWithAtoms(subdomain, 100000, 1_r, 1_r);

    // set up ghost layer for periodic boundary conditions
    communication::GhostLayer ghostLayer;

    // set up timer for runtime measurement
    Kokkos::Timer timer;

    // main simulation loop
    for (auto i = 0; i < config.nsteps; ++i)
    {
        // integrate equations of motion before (potentially) calculating forces
        action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        // reinsert atoms that left the domain according to periodic boundary conditions
        ghostLayer.exchangeRealAtoms(atoms, subdomain);

        // finish integrating equations of motion
        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        // handle output
        if (i % 100 == 0)
        {
            io::dumpCSV("atoms_" + std::to_string(i) + ".csv", atoms);
        }
    }
    // print performance data
    auto time = timer.seconds();
    std::cout << time << std::endl;
}

int main(int argc, char* argv[])  // NOLINT
{
    // initialize Kokkos environment
    Kokkos::ScopeGuard scope_guard(argc, argv);

    // print Kokkos execution space
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    // initialize simulation configuration with command line interface
    Config config;
    CLI::App app{"ideal gas benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    CLI11_PARSE(app, argc, argv);

    // run simulation of the ideal gas
    runIdealGas(config);

    return EXIT_SUCCESS;
}