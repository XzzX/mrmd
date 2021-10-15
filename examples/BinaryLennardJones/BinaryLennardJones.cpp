#include <fmt/format.h>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/VelocityScaling.hpp"
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
    bool bOutput = true;
    idx_t outputInterval = -1;

    idx_t nsteps = 40001;
    real_t dt = 0.001_r;

    real_t sigma = 1_r;
    real_t epsilon = 1_r;
    bool isShifted = true;
    real_t rc = 2.5;

    real_t temperature = 1.02_r;
    real_t gamma = 1_r;

    real_t Lx = 36_r * sigma;
    real_t Ly = 5_r * sigma;
    real_t Lz = 5_r * sigma;
    idx_t numParticles = 3600;
    real_t fracTypeA = 0.8_r;

    real_t cell_ratio = 1.0_r;
    idx_t estimatedMaxNeighbors = 60;
    real_t skin = 0.3;
    real_t neighborCutoff = rc + skin;
};

data::Atoms fillDomainWithAtomsSC(const data::Subdomain& subdomain,
                                  const idx_t& numAtoms,
                                  const real_t& fracTypeA,
                                  const real_t& maxVelocity)
{
    assert(fracTypeA < 1_r);
    assert(fracTypeA > 0_r);
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    data::Atoms atoms(numAtoms);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto mass = atoms.getMass();
    auto type = atoms.getType();

    auto numAtomsA = int_c(real_c(numAtoms) * fracTypeA);

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

        type(idx) = idx < numAtomsA ? 0 : 1;
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

void LJ(Config& config)
{
    auto subdomain =
        data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Ly, config.Lz}, config.neighborCutoff);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numParticles, config.fracTypeA, 1_r);
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    communication::GhostLayer ghostLayer(subdomain);
    std::vector<real_t> sigma = {
        config.sigma, 0.80_r * config.sigma, 0.80_r * config.sigma, 0.88_r * config.sigma};
    std::vector<real_t> epsilon = {
        config.epsilon, 1.50_r * config.epsilon, 1.50_r * config.epsilon, 0.50_r * config.epsilon};
    std::vector<real_t> cappingDistance(4);
    std::transform(
        sigma.begin(), sigma.end(), cappingDistance.begin(), [&](auto& v) { return 0.8_r * v; });
    std::vector<real_t> rc(4);
    std::transform(sigma.begin(), sigma.end(), rc.begin(), [&](auto& v) { return config.rc * v; });

    action::LennardJones LJ(cappingDistance, rc, sigma, epsilon, 2, config.isShifted);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    action::VelocityScaling velocityScaling(0.5_r, config.temperature);
    VerletList verletList;
    Kokkos::Timer timer;
    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    util::printTable("step", "time", "T", "Ek", "E0", "E", "p", "Nlocal", "Nghost");
    util::printTableSep("step", "time", "T", "Ek", "E0", "E", "p", "Nlocal", "Nghost");
    util::ExponentialMovingAverage pressure(0.05_r);
    for (auto step = 0; step < config.nsteps; ++step)
    {
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        if (maxAtomDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            real_t gridDelta[3] = {
                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            LinkedCellList linkedCellList(atoms.getPos(),
                                          0,
                                          atoms.numLocalAtoms,
                                          gridDelta,
                                          subdomain.minCorner.data(),
                                          subdomain.maxCorner.data());
            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(atoms);
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

        LJ.applyForces(atoms, verletList);
        if ((config.temperature >= 0) && (step % 100 == 0))
        {
            velocityScaling.apply(atoms, 3_r * real_c(atoms.numLocalAtoms));
        }
        pressure << analysis::getPressure(atoms, subdomain);
        ghostLayer.contributeBackGhostToReal(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto E0 = LJ.computeEnergy(atoms, verletList);
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto T = (2_r / 3_r) * Ek;
            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             pressure,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            io::dumpCSV(fmt::format("spc_{:0>6}.csv", step), atoms, false);
        }
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"Binary Lennard Jones Fluid benchmark application"};
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