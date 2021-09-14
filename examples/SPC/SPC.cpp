#include "action/SPC.hpp"

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "action/ContributeMoleculeForceToAtoms.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/ThermodynamicForce.hpp"
#include "action/UpdateMolecules.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/SystemMomentum.hpp"
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
    idx_t nsteps = 500001;
    real_t dt = 0.002_r;  ///< unit: ps

    // simulation box parameters
    real_t rho = 32_r;  ///< unit: molecules/nm**3

    // thermodynamic force parameters
    real_t thermodynamicForceModulation = 2_r;

    // neighborlist parameters
    real_t skin = 0.3_r;  ///< unit: nm
    real_t neighborCutoff = action::SPC::rc + skin;
    real_t cell_ratio = 0.5_r;
    idx_t estimatedMaxNeighbors = 60;

    // thermostat parameters
    real_t temperature = 2.47_r;  ///< unit: kJ/k_b/mol
    real_t gamma = 10_r;

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

void initMolecules(data::Molecules& molecules,
                   data::Particles& atoms,
                   data::Subdomain& subdomain,
                   idx_t numMolecules)
{
    molecules.resize(numMolecules * 2);
    auto atomsOffset = molecules.getAtomsOffset();
    auto numAtoms = molecules.getNumAtoms();
    auto moleculesPolicy = Kokkos::RangePolicy<>(0, numMolecules);
    auto moleculesKernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        atomsOffset(idx) = idx * 3;
        numAtoms(idx) = 3;
    };
    Kokkos::parallel_for("initMolecules", moleculesPolicy, moleculesKernel);
    molecules.numLocalMolecules = numMolecules;
    molecules.numGhostMolecules = 0;

    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);
    atoms.resize(numMolecules * 3 * 2);
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto charge = atoms.getCharge();
    auto type = atoms.getType();
    auto mass = atoms.getMass();
    auto realtiveMass = atoms.getRelativeMass();
    auto atomPolicy = Kokkos::RangePolicy<>(0, numMolecules);
    auto atomKernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto randGen = RNG.get_state();
        // oxygen
        pos(idx * 3 + 0, 0) = randGen.drand() * subdomain.diameter[0] + subdomain.minCorner[0];
        pos(idx * 3 + 0, 1) = randGen.drand() * subdomain.diameter[1] + subdomain.minCorner[1];
        pos(idx * 3 + 0, 2) = randGen.drand() * subdomain.diameter[2] + subdomain.minCorner[2];

        //        vel(idx * 3 + 0, 0) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 0, 1) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 0, 2) = (randGen.drand() - 0.5_r) * 1_r;

        type(idx * 3 + 0) = 0;
        mass(idx * 3 + 0) = 15.999_r;  ///< unit: g/mol
        charge(idx * 3 + 0) = -0.82_r;
        realtiveMass(idx * 3 + 0) = 15.999_r / (15.999_r + 2_r * 1.008_r);

        // hydrogen 1
        pos(idx * 3 + 1, 0) = pos(idx * 3 + 0, 0) + action::SPC::eqDistanceHO;
        pos(idx * 3 + 1, 1) = pos(idx * 3 + 0, 1);
        pos(idx * 3 + 1, 2) = pos(idx * 3 + 0, 2);

        //        vel(idx * 3 + 1, 0) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 1, 1) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 1, 2) = (randGen.drand() - 0.5_r) * 1_r;

        type(idx * 3 + 1) = 1;
        mass(idx * 3 + 1) = 1.008_r;  ///< unit: g/mol
        charge(idx * 3 + 1) = +0.41_r;
        realtiveMass(idx * 3 + 1) = 1.008_r / (15.999_r + 2_r * 1.008_r);

        // hydrogen 2
        pos(idx * 3 + 2, 0) =
            pos(idx * 3 + 0, 0) + action::SPC::eqDistanceHO * std::cos(action::SPC::angleHOH);
        pos(idx * 3 + 2, 1) =
            pos(idx * 3 + 0, 1) + action::SPC::eqDistanceHO * std::sin(action::SPC::angleHOH);
        pos(idx * 3 + 2, 2) = pos(idx * 3 + 0, 2);

        //        vel(idx * 3 + 2, 0) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 2, 1) = (randGen.drand() - 0.5_r) * 1_r;
        //        vel(idx * 3 + 2, 2) = (randGen.drand() - 0.5_r) * 1_r;

        type(idx * 3 + 2) = 1;
        mass(idx * 3 + 2) = 1.008_r;  ///< unit: g/mol
        charge(idx * 3 + 2) = +0.41_r;
        realtiveMass(idx * 3 + 2) = 1.008_r / (15.999_r + 2_r * 1.008_r);

        // Give the state back, which will allow another thread to acquire it
        RNG.free_state(randGen);
    };
    Kokkos::parallel_for("initAtoms", atomPolicy, atomKernel);
    atoms.numLocalParticles = numMolecules * 3;
    atoms.numGhostParticles = 0;
}

void SPC(Config& config)
{
    auto subdomain = data::Subdomain({0_r, 0_r, 0_r}, {5_r, 5_r, 5_r}, config.neighborCutoff);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];

    data::Particles atoms(0);
    data::Molecules molecules(0);
    initMolecules(molecules, atoms, subdomain, idx_c(volume * config.rho));
    io::dumpCSV("particles_initial.csv", atoms);

    std::cout << "molecules added: " << molecules.numLocalMolecules << std::endl;

    auto rho = real_c(molecules.numLocalMolecules) / volume;
    std::cout << "global molecule density: " << rho << std::endl;

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
    action::SPC spc;
    action::ThermodynamicForce thermodynamicForce(
        config.rho, subdomain, config.thermodynamicForceModulation);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    communication::MultiResGhostLayer ghostLayer(subdomain);

    util::printTable("step", "time", "T", "Ek", "E0", "E", "mu", "Nlocal", "Nghost");
    util::printTableSep("step", "time", "T", "Ek", "E0", "E", "mu", "Nlocal", "Nghost");
    for (auto step = 0; step < config.nsteps; ++step)
    {
        assert(atoms.numLocalParticles == molecules.numLocalMolecules * 3);
        assert(atoms.numGhostParticles == molecules.numGhostMolecules * 3);
        spc.enforceConstraints(molecules, atoms, config.dt);
        maxParticleDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        // update molecule positions
        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        if (maxParticleDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxParticleDisplacement = 0_r;

            ghostLayer.exchangeRealParticles(molecules, atoms);

            //            real_t gridDelta[3] = {
            //                config.neighborCutoff, config.neighborCutoff,
            //    config.neighborCutoff
            //};
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

        //        if (step % config.densitySamplingInterval == 0)
        //        {
        //            thermodynamicForce.sample(atoms);
        //        }
        //
        //        if (step % config.densityUpdateInterval == 0)
        //        {
        //            thermodynamicForce.update();
        //        }

        //        thermodynamicForce.apply(atoms);
        auto E0 = 0_r;
        E0 += spc.applyForces(molecules, moleculesVerletList, atoms);
        action::ContributeMoleculeForceToAtoms::update(molecules, atoms);
        if (config.temperature >= 0)
        {
            langevinThermostat.apply(atoms);
        }
        ghostLayer.contributeBackGhostToReal(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto Ek = analysis::getKineticEnergy(atoms) / real_c(atoms.numLocalParticles);
            auto T = Ek;
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

            io::dumpCSV("particles_" + std::to_string(step) + ".csv", atoms, false);

            //            fThermodynamicForceOut << thermodynamicForce.getForce() << std::endl;
            //            fDriftForceCompensation << LJ.getMeanCompensationEnergy() << std::endl;
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
    SPC(config);

    return EXIT_SUCCESS;
}