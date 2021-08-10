#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "action/ContributeMoleculeForceToAtoms.hpp"
#include "action/LJ_IdealGas.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/ThermodynamicForce.hpp"
#include "action/UpdateMolecules.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/SystemMomentum.hpp"
#include "analysis/Temperature.hpp"
#include "communication/MultiResGhostLayer.hpp"
#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"
#include "io/RestoreLAMMPS.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/PrintTable.hpp"
#include "util/Random.hpp"
#include "weighting_function/Slab.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = true;
    idx_t outputInterval = -1;

    // general simulation parameters
    idx_t nsteps = 5000001;
    real_t dt = 0.0005_r;

    // simulation box parameters
    real_t rho = 0.86_r;

    // thermodynamic force parameters
    real_t thermodynamicForceModulation = 2_r;

    // LJ parameters
    real_t sigma = 1_r;
    real_t epsilon = 1_r;
    real_t rc = 2.5_r;

    // neighborlist parameters
    real_t skin = 0.3_r;
    real_t neighborCutoff = rc + skin;
    real_t cell_ratio = 0.5_r;
    idx_t estimatedMaxNeighbors = 60;

    // thermostat parameters
    real_t temperature = 1.2_r;
    real_t gamma = 1_r;

    // AdResS parameters
    real_t atomisticRegionDiameter = 10_r;
    real_t hybridRegionDiameter = 2.5_r;
    idx_t lambdaExponent = 7;
    idx_t DriftForceSamplingInterval = 200;
    idx_t DriftForceUpdateInterval = 20000;
    real_t DriftForceBinSize = 0.005_r;

    idx_t densitySamplingInterval = 200;
    idx_t densityUpdateInterval = 50000;
    real_t DensityBinSize = 0.5_r;
    real_t convSigma = 2_r;
    real_t convRange = 2_r;
};

void LJ(Config& config)
{
    auto subdomain = data::Subdomain(
        {-2.5038699752178008e+01_r, -8.3462332507260033e+00_r, -8.3462332507260033e+00_r},
        {2.5038699752178008e+01, 8.3462332507260033e+00, 8.3462332507260033e+00},
        config.neighborCutoff);

    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    const idx_t numParticles = idx_c(config.rho * volume);
    util::Random RNG;
    data::Particles atoms(numParticles * 2);
    data::Molecules molecules(numParticles * 2);
    io::restoreLAMMPS("LJ_spartian_3.lammpstrj", atoms, molecules);
    std::cout << "particles added: " << atoms.numLocalParticles << std::endl;
    std::cout << "system temperature: " << analysis::getTemperature(atoms) << std::endl;

    auto rho = real_c(atoms.numLocalParticles) / volume;
    std::cout << "global particle density: " << rho << std::endl;

    // data allocations
    VerletList moleculesVerletList;
    idx_t verletlistRebuildCounter = 0;

    Kokkos::Timer timer;
    real_t maxParticleDisplacement = std::numeric_limits<real_t>::max();
    auto weightingFunction = weighting_function::Slab({0_r, 0_r, 0_r},
                                                      config.atomisticRegionDiameter,
                                                      config.hybridRegionDiameter,
                                                      config.lambdaExponent);
    std::ofstream fDensityOut("densityProfile.txt");
    std::ofstream fThermodynamicForceOut("thermodynamicForce.txt");
    std::ofstream fDriftForceCompensation("driftForce.txt");

    // actions
    action::LJ_IdealGas LJ(0.1_r, config.rc, config.sigma, config.epsilon, true);
    action::ThermodynamicForce thermodynamicForce(
        config.rho, subdomain, config.thermodynamicForceModulation);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    communication::MultiResGhostLayer ghostLayer(subdomain);

    util::printTable("step", "time", "T", "Ek", "E0", "E", "mu", "Nlocal", "Nghost");
    util::printTableSep("step", "time", "T", "Ek", "E0", "E", "mu", "Nlocal", "Nghost");
    for (auto step = 0; step < config.nsteps; ++step)
    {
        assert(atoms.numLocalParticles == molecules.numLocalMolecules);
        assert(atoms.numGhostParticles == molecules.numGhostMolecules);
        maxParticleDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        // update molecule positions
        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        if (maxParticleDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxParticleDisplacement = 0_r;

            ghostLayer.exchangeRealParticles(molecules, atoms);

            //            real_t gridDelta[3] = {
            //                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            //            LinkedCellList linkedCellList(atoms.getPos(),
            //                                          0,
            //                                          atoms.numLocalParticles,
            //                                          gridDelta,
            //                                          subdomain.minCorner.data(),
            //                                          subdomain.maxCorner.data());
            //            particles.permute(linkedCellList);

            ghostLayer.createGhostParticles(molecules, atoms);
            moleculesVerletList.build(molecules.getPos(),
                                      0,
                                      molecules.numLocalMolecules,
                                      config.neighborCutoff,
                                      config.cell_ratio,
                                      subdomain.minGhostCorner.data(),
                                      subdomain.maxGhostCorner.data(),
                                      config.estimatedMaxNeighbors);
            ++verletlistRebuildCounter;
        }
        else
        {
            ghostLayer.updateGhostParticles(atoms);
        }

        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        auto atomsForce = atoms.getForce();
        Cabana::deep_copy(atomsForce, 0_r);
        auto moleculesForce = molecules.getForce();
        Cabana::deep_copy(moleculesForce, 0_r);

        if (step % config.densitySamplingInterval == 0)
        {
            thermodynamicForce.sample(atoms);
        }

        if (step % config.densityUpdateInterval == 0)
        {
            thermodynamicForce.update();
        }

        thermodynamicForce.apply(atoms);
        auto E0 = LJ.run(molecules, moleculesVerletList, atoms);
        action::ContributeMoleculeForceToAtoms::update(molecules, atoms);
        if (config.temperature >= 0)
        {
            langevinThermostat.apply(atoms);
        }
        ghostLayer.contributeBackGhostToReal(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto T = analysis::getTemperature(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto Ek = (3_r / 2_r) * T;
            E0 /= real_c(atoms.numLocalParticles);

            // calc chemical potential
            auto Fth = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           thermodynamicForce.getForce().data);
            auto mu = 0_r;
            for (auto i = 0; i < Fth.extent(0) / 2; ++i)
            {
                mu += Fth(i);
            }
            mu *= thermodynamicForce.getForce().binSize;

            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             mu,
                             atoms.numLocalParticles,
                             atoms.numGhostParticles);

            io::dumpCSV("particles_" + std::to_string(step) + ".csv", atoms);

            fThermodynamicForceOut << thermodynamicForce.getForce() << std::endl;
            fDriftForceCompensation << LJ.getMeanCompensationEnergy() << std::endl;
        }
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;
    fDensityOut.close();
    fThermodynamicForceOut.close();
    fDriftForceCompensation.close();

    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalParticles << ", " << config.nsteps
         << std::endl;
    fout.close();
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"AdResS LJ-IG benchmark simulation"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option("-o,--output", config.outputInterval, "output interval");
    CLI11_PARSE(app, argc, argv);

    if (config.outputInterval < 0) config.bOutput = false;
    LJ(config);

    return EXIT_SUCCESS;
}