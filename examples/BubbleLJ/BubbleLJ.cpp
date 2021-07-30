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
#include "analysis/AxialDensityProfile.hpp"
#include "analysis/SmoothenDensityProfile.hpp"
#include "analysis/SystemMomentum.hpp"
#include "analysis/Temperature.hpp"
#include "communication/MultiResGhostLayer.hpp"
#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"
#include "util/Random.hpp"
#include "weighting_function/Slab.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = true;

    // general simulation parameters
    idx_t nsteps = 100000;
    real_t dt = 0.0005_r;

    // simulation box parameters
    real_t Lx = 30_r;
    real_t Ly = 10_r;
    real_t Lz = 10_r;
    real_t rho = 0.86_r;

    // thermodynamic force parameters
    real_t thermodynamicForceModulation = 1_r;

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
    idx_t densitySamplingInterval = 10;
    idx_t spartianInterval = 100;
    real_t atomisticRegionDiameter = 10_r;
    real_t hybridRegionDiameter = 2_r;
    idx_t lambdaExponent = 7;
};

void LJ(Config& config)
{
    auto subdomain =
        data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Ly, config.Lz}, config.neighborCutoff);

    const auto volume = config.Lx * config.Ly * config.Lz;
    const idx_t numParticles = idx_c(config.rho * volume);
    util::Random RNG;
    data::Particles atoms(numParticles * 2);
    data::Molecules molecules(numParticles * 2);
    for (auto idx = 0; idx < numParticles; ++idx)
    {
        atoms.getPos()(idx, 0) = RNG.draw() * config.Lx;
        atoms.getPos()(idx, 1) = RNG.draw() * config.Ly;
        atoms.getPos()(idx, 2) = RNG.draw() * config.Lz;

        atoms.getVel()(idx, 0) = (RNG.draw() - 0.5_r) * 2_r;
        atoms.getVel()(idx, 1) = (RNG.draw() - 0.5_r) * 2_r;
        atoms.getVel()(idx, 2) = (RNG.draw() - 0.5_r) * 2_r;

        atoms.getRelativeMass()(idx) = 1_r;

        molecules.getAtomsEndIdx()(idx) = idx + 1;
    }
    atoms.numLocalParticles = numParticles;
    molecules.numLocalMolecules = numParticles;
    std::cout << "particles added: " << numParticles << std::endl;

    auto rho = real_c(atoms.numLocalParticles) / volume;
    std::cout << "global particle density: " << rho << std::endl;

    // data allocations
    VerletList moleculesVerletList;
    idx_t verletlistRebuildCounter = 0;
    data::Histogram densityProfile("density-profile", 0_r, config.Lx, 100);
    idx_t densityProfileEvaluations = 0;
    data::Histogram thermodynamicForce("thermodynamic-force", 0_r, config.Lx, 100);
    Kokkos::Timer timer;
    real_t maxParticleDisplacement = std::numeric_limits<real_t>::max();
    auto weightingFunction =
        weighting_function::Slab({0.5_r * config.Lx, 0.5_r * config.Ly, 0.5_r * config.Lz},
                                 config.atomisticRegionDiameter,
                                 config.hybridRegionDiameter,
                                 config.lambdaExponent);
    std::ofstream fDensityOut("densityProfile.txt");
    std::ofstream fThermodynamicForceOut("thermodynamicForce.txt");

    // actions
    action::LennardJones LJ(config.rc, config.sigma, config.epsilon);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    communication::MultiResGhostLayer ghostLayer(subdomain);

    for (auto i = 0; i < config.nsteps; ++i)
    {
        assert(atoms.numLocalParticles == molecules.numLocalMolecules);
        assert(atoms.numGhostParticles == molecules.numGhostMolecules);
        maxParticleDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

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

        auto atomsForce = atoms.getForce();
        Cabana::deep_copy(atomsForce, 0_r);
        auto moleculesForce = molecules.getForce();
        Cabana::deep_copy(moleculesForce, 0_r);

        if (i % config.densitySamplingInterval == 0)
        {
            densityProfile += analysis::getAxialDensityProfile(
                atoms.getPos(), atoms.numLocalParticles, 0_r, config.Lx, 100);
            ++densityProfileEvaluations;
            if (densityProfileEvaluations == 1)
            {
                for (auto i = 0; i < thermodynamicForce.numBins; ++i)
                {
                    fDensityOut << densityProfile.data(i) << " ";
                }
                fDensityOut << std::endl;
            }
        }

        if (i % config.spartianInterval == 0)
        {
            auto binVolume = config.Ly * config.Lz * densityProfile.binSize;
            auto normalizationFactor = 1_r / (binVolume * real_c(densityProfileEvaluations)) *
                                       config.thermodynamicForceModulation / rho;
            auto policy = Kokkos::RangePolicy<>(0, densityProfile.numBins);
            Kokkos::parallel_for(
                policy, KOKKOS_LAMBDA(const idx_t idx) {
                    densityProfile.data(idx) *= normalizationFactor;
                });
            Kokkos::fence();
            auto smoothedDensityProfile =
                analysis::smoothenDensityProfile(densityProfile, 3_r, 6_r);

            thermodynamicForce += data::gradient(smoothedDensityProfile);

            Kokkos::deep_copy(densityProfile.data, 0_r);
            densityProfileEvaluations = 0;
        }

        action::ThermodynamicForce::apply(atoms, thermodynamicForce);
        action::LJ_IdealGas::applyForces(config.sigma * 0.5_r,
                                         config.rc,
                                         config.sigma,
                                         config.epsilon,
                                         molecules,
                                         moleculesVerletList,
                                         atoms,
                                         true);
        action::ContributeMoleculeForceToAtoms::update(molecules, atoms);
        if (config.temperature >= 0)
        {
            langevinThermostat.applyThermostat(atoms);
        }
        ghostLayer.contributeBackGhostToReal(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (i % config.spartianInterval == 0))
        {
            //            VerletList atomsVerletList(atoms.getPos(),
            //                                       0,
            //                                       atoms.numLocalParticles,
            //                                       config.neighborCutoff,
            //                                       config.cell_ratio,
            //                                       subdomain.minGhostCorner.data(),
            //                                       subdomain.maxGhostCorner.data(),
            //                                       config.estimatedMaxNeighbors);
            //            auto E0 = 0_r;  // LJ.computeEnergy(atoms, atomsVerletList);
            //            auto T = analysis::getTemperature(atoms);
            //            auto systemMomentum = analysis::getSystemMomentum(atoms);
            //            auto Ek = (3_r / 2_r) * real_c(atoms.numLocalParticles) * T;
            std::cout << i << ": " << timer.seconds() << std::endl;
            //            std::cout << "system momentum: " << systemMomentum[0] << " | " <<
            //            systemMomentum[1]
            //                      << " | " << systemMomentum[2] << std::endl;
            //            std::cout << "rebuild counter: " << verletlistRebuildCounter << std::endl;
            //            std::cout << "T : " << std::setw(10) << T << " | ";
            //            std::cout << "Ek: " << std::setw(10) << Ek << " | ";
            //            std::cout << "E0: " << std::setw(10) << E0 << " | ";
            //            std::cout << "E : " << std::setw(10) << E0 + Ek << " | ";
            //            std::cout << "Nlocal : " << std::setw(10) << atoms.numLocalParticles << "
            //            | "; std::cout << "Nghost : " << std::setw(10) << atoms.numGhostParticles
            //            << std::endl;

            //            io::dumpCSV("particles_" + std::to_string(i) + ".csv", atoms);

            for (auto i = 0; i < thermodynamicForce.numBins; ++i)
            {
                fThermodynamicForceOut << thermodynamicForce.data(i) << " ";
            }
            fThermodynamicForceOut << std::endl;
        }
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;
    fDensityOut.close();
    fThermodynamicForceOut.close();

    auto cores = std::getenv("OMP_NUM_THREADS") != nullptr
                     ? std::string(std::getenv("OMP_NUM_THREADS"))
                     : std::string("0");

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << atoms.numLocalParticles << ", " << config.nsteps
         << std::endl;
    fout.close();

    //    auto E0 = LJ.computeEnergy(atoms, verletList);
    auto T = analysis::getTemperature(atoms);

    //    CHECK_LESS(E0, -162000_r);
    //    CHECK_GREATER(E0, -163000_r);
    //
    //    CHECK_LESS(T, 1.43_r);
    //    CHECK_GREATER(T, 1.41_r);
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"Lennard Jones Fluid benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    app.add_option(
        "-T,--temperature",
        config.temperature,
        "temperature of the Langevin thermostat (negative numbers deactivate the thermostat)");
    app.add_flag("-o,--output", config.bOutput, "print physical state regularly");
    CLI11_PARSE(app, argc, argv);

    LJ(config);

    return EXIT_SUCCESS;
}