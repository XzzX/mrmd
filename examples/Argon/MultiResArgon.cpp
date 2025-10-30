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

#include <format>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/BerendsenThermostat.hpp"
#include "action/ContributeMoleculeForceToAtoms.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/LimitAcceleration.hpp"
#include "action/LimitVelocity.hpp"
#include "action/UpdateMolecules.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/MeanSquareDisplacement.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/MultiResGhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/MoleculesFromAtoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"
#include "io/DumpGRO.hpp"
#include "io/RestoreTXT.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/PrintTable.hpp"
#include "weighting_function/Slab.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = true;
    idx_t outputInterval = -1;

    idx_t nsteps = 400001;
    static constexpr idx_t numAtoms = 16 * 16 * 16;

    static constexpr real_t sigma = 0.34_r;        ///< units: nm
    static constexpr real_t epsilon = 0.993653_r;  ///< units: kJ/mol
    static constexpr real_t mass = 39.948_r;       ///< units: u

    static constexpr real_t rc = 2.3_r * sigma;
    static constexpr real_t skin = 0.1;
    static constexpr real_t neighborCutoff = rc + skin;

    static constexpr real_t dt = 0.00217;  ///< units: ps
    real_t temperature = 3.0_r;
    real_t gamma = 0.7_r / dt;

    real_t Lx = 3.196 * 2_r;  ///< units: nm

    real_t cell_ratio = 1.0_r;

    idx_t estimatedMaxNeighbors = 60;

    const std::string resName = "Argon"; 
    const std::vector<std::string> typeNames = {"Ar"};

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
    auto relativeMass = atoms.getRelativeMass();

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
        relativeMass(idx) = 1_r;
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

void LJ(Config& config)
{
    auto subdomain =
        data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Lx, config.Lx}, config.neighborCutoff);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numAtoms, 1_r);
    auto molecules = data::createMoleculeForEachAtom(atoms);
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    io::dumpGRO("atoms_initial.gro", atoms, subdomain, 0_r, "Argon", config.resName, config.typeNames, false);

    communication::MultiResGhostLayer ghostLayer;
    weighting_function::Slab weightingFunction({-100_r, -100_r, -100_r}, 1_r, 1_r, 1);
    action::LennardJones LJ(config.rc, config.sigma, config.epsilon, 0.7_r * config.sigma);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    analysis::MeanSquareDisplacement meanSquareDisplacement;
    meanSquareDisplacement.reset(atoms);
    auto msd = 0_r;
    HalfVerletList verletList;
    Kokkos::Timer timer;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    util::printTable("step", "time", "T", "Ek", "E0", "E", "p", "msd", "Nlocal", "Nghost");
    util::printTableSep("step", "time", "T", "Ek", "E0", "E", "p", "msd", "Nlocal", "Nghost");

    std::ofstream fStat("statistics.txt");
    for (auto step = 0; step < config.nsteps; ++step)
    {
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        if (maxAtomDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(molecules, atoms, subdomain);

            //            real_t gridDelta[3] = {
            //                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            //            LinkedCellList linkedCellList(atoms.getPos(),
            //                                          0,
            //                                          atoms.numLocalAtoms,
            //                                          gridDelta,
            //                                          subdomain.minCorner.data(),
            //                                          subdomain.maxCorner.data());
            //            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(molecules, atoms, subdomain);
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

        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);
        auto moleculesForce = molecules.getForce();
        Cabana::deep_copy(moleculesForce, 0_r);

        LJ.apply(atoms, verletList);
        action::ContributeMoleculeForceToAtoms::update(molecules, atoms);

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

            fStat << step << " " << timer.seconds() << " " << T << " " << Ek << " " << E0 << " "
                  << E0 + Ek << " " << p << " " << msd << " " << atoms.numLocalAtoms << " "
                  << atoms.numGhostAtoms << " " << std::endl;

            io::dumpGRO(std::format("argon_{:0>6}.gro", step),
                        atoms,
                        subdomain,
                        step * config.dt,
                        "Argon",
                        config.resName, 
                        config.typeNames,
                        false);

            //            io::dumpCSV("atoms_" + std::to_string(step) + ".csv", atoms,
            //            false);
        }

        if (step % 1000 == 0)
        {
            msd = meanSquareDisplacement.calc(atoms, subdomain) / (1000_r * config.dt);
            if ((config.temperature > 0_r) && (step > 5000))
            {
                config.temperature -= 7.8e-3_r;
                config.temperature = std::max(config.temperature, 0_r);
            }

            langevinThermostat.set(config.gamma, config.temperature * 0.5_r, config.dt);
            langevinThermostat.apply(atoms);

            meanSquareDisplacement.reset(atoms);
        }

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);
    }
    fStat.close();
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