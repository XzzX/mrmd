#include "action/LennardJones.hpp"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/LangevinThermostat.hpp"
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
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    std::string filename = "";
    bool bOutput = true;
    idx_t outputInterval = -1;

    idx_t nsteps = 2001;
    real_t rc = 2.5;
    real_t skin = 0.3;
    real_t neighborCutoff = rc + skin;
    real_t sigma = real_t(1);
    real_t epsilon = real_t(1);
    real_t dt = 0.001;
    real_t temperature = real_t(1.12);
    real_t gamma = real_t(1);

    real_t Lx = 33.8585;
    real_t rho = real_t(0.75);

    real_t cell_ratio = real_t(1.0);

    idx_t estimatedMaxNeighbors = 60;
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

        vel(idx, 0) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        vel(idx, 1) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        vel(idx, 2) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        RNG.free_state(randGen);

        mass(idx) = real_t(1);
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

void LJ(Config& config)
{
    auto subdomain = data::Subdomain({real_t(0), real_t(0), real_t(0)},
                                     {config.Lx, config.Lx, config.Lx},
                                     config.neighborCutoff);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    data::Atoms atoms(0);

    if (!config.filename.empty())
    {
        atoms = io::restoreAtoms(config.filename);
    }
    else
    {
        atoms = fillDomainWithAtomsSC(subdomain, idx_c(config.rho * volume), real_t(1));
    }

    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    communication::GhostLayer ghostLayer;
    action::LennardJones LJ(config.rc, config.sigma, config.epsilon, real_t(0.7) * config.sigma);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    HalfVerletList verletList;
    Kokkos::Timer timer;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    util::printTable("step", "time", "T", "Ek", "E0", "E", "p", "p2", "Nlocal", "Nghost");
    util::printTableSep("step", "time", "T", "Ek", "E0", "E", "p", "p2", "Nlocal", "Nghost");
    for (auto step = 0; step < config.nsteps; ++step)
    {
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        if (maxAtomDisplacement >= config.skin * real_t(0.5))
        {
            // reset displacement
            maxAtomDisplacement = real_t(0);

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            real_t gridDelta[3] = {
                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            LinkedCellList linkedCellList(atoms.getPos(),
                                          0,
                                          atoms.numLocalAtoms,
                                          gridDelta,
                                          subdomain.minCorner.data(),
                                          subdomain.maxCorner.data());
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

        auto force = atoms.getForce();
        Cabana::deep_copy(force, real_t(0));

        LJ.apply(atoms, verletList);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto E0 = LJ.getEnergy() / real_c(atoms.numLocalAtoms);
            auto Ek = analysis::getKineticEnergy(atoms);
            auto p2 = real_t(2) * (Ek - LJ.getVirial()) / (real_t(3) * volume);
            Ek /= real_c(atoms.numLocalAtoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (real_t(2) / real_t(3)) * Ek;
            auto p = analysis::getPressure(atoms, subdomain);

            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             p,
                             p2,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            io::dumpCSV("lj_" + std::to_string(step) + ".csv", atoms);
        }

        if ((config.temperature >= 0) && (step < 10000))
        {
            langevinThermostat.apply(atoms);
        }

        ghostLayer.contributeBackGhostToReal(atoms);
        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);
    }
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
    app.add_option("-f,--filename", config.filename, "input file");
    CLI11_PARSE(app, argc, argv);
    if (config.outputInterval < 0) config.bOutput = false;
    LJ(config);

    return EXIT_SUCCESS;
}