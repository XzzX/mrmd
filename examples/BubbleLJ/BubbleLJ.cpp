#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/LJ_IdealGas.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
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
#include "io/RestoreTXT.hpp"
#include "util/Random.hpp"
#include "weighting_function/Spherical.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = true;

    idx_t nsteps = 2001;
    real_t sigma = 1_r;
    real_t epsilon = 1_r;
    real_t rc = 2.5;
    real_t skin = 0.3;
    real_t neighborCutoff = rc + skin;
    real_t dt = 0.005;
    real_t temperature = 1_r;
    real_t gamma = 1_r;

    real_t Lx = 55_r;

    real_t cell_ratio = 0.5_r;

    idx_t estimatedMaxNeighbors = 60;
};

void LJ(Config& config)
{
    auto subdomain = data::Subdomain({-config.Lx, -config.Lx, -config.Lx},
                                     {config.Lx, config.Lx, config.Lx},
                                     config.neighborCutoff);

    util::Random RNG;
    data::Particles atoms(100 * 100 * 2);
    data::Molecules molecules(100 * 100 * 2);
    idx_t idx = 0;
    for (auto x = -50; x < 51; ++x)
    {
        for (auto y = -50; y < 51; ++y)
        {
            atoms.getPos()(idx, 0) = real_c(x);
            atoms.getPos()(idx, 1) = real_c(y);
            atoms.getPos()(idx, 2) = 0_r;

            atoms.getVel()(idx, 0) = (RNG.draw() - 0.5_r) * 2_r;
            atoms.getVel()(idx, 1) = (RNG.draw() - 0.5_r) * 2_r;
            atoms.getVel()(idx, 2) = 0_r;

            atoms.getRelativeMass()(idx) = 1_r;

            molecules.getAtomsEndIdx()(idx) = idx + 1;
            ++idx;
        }
    }
    atoms.numLocalParticles = idx;
    molecules.numLocalMolecules = idx;
    std::cout << "particles added: " << idx << std::endl;

    communication::MultiResGhostLayer ghostLayer(subdomain);
    action::LennardJones LJ(config.rc, config.sigma, config.epsilon);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    VerletList moleculesVerletList;
    Kokkos::Timer timer;
    real_t maxParticleDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    auto weightingFunction = weighting_function::Spherical({0_r, 0_r, 0_r}, 20_r, 10_r, 7);
    for (auto i = 0; i < config.nsteps; ++i)
    {
        std::cout << i << " | " << molecules.numGhostMolecules << std::endl;
        assert(atoms.numLocalParticles == molecules.numLocalMolecules);
        assert(atoms.numGhostParticles == molecules.numGhostMolecules);
        maxParticleDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, config.dt);

        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        if (maxParticleDisplacement >= -config.skin * 0.5_r)
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
            ++rebuildCounter;
        }
        else
        {
            ghostLayer.updateGhostParticles(atoms);
        }

        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        //        LJ.applyForces(atoms, moleculesVerletList);
        action::LJ_IdealGas::applyForces(
            config.rc, config.sigma, config.epsilon, molecules, moleculesVerletList, atoms);
        if (config.temperature >= 0)
        {
            langevinThermostat.applyThermostat(atoms);
        }
        ghostLayer.contributeBackGhostToReal(atoms);

        action::VelocityVerlet::postForceIntegrate(atoms, config.dt);

        if (config.bOutput && (i % 10 == 0))
        {
            //            VerletList atomsVerletList(atoms.getPos(),
            //                                       0,
            //                                       atoms.numLocalParticles,
            //                                       config.neighborCutoff,
            //                                       config.cell_ratio,
            //                                       subdomain.minGhostCorner.data(),
            //                                       subdomain.maxGhostCorner.data(),
            //                                       config.estimatedMaxNeighbors);
            auto E0 = 0_r;  // LJ.computeEnergy(atoms, atomsVerletList);
            auto T = analysis::getTemperature(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto Ek = (3_r / 2_r) * real_c(atoms.numLocalParticles) * T;
            std::cout << i << ": " << timer.seconds() << std::endl;
            std::cout << "system momentum: " << systemMomentum[0] << " | " << systemMomentum[1]
                      << " | " << systemMomentum[2] << std::endl;
            std::cout << "rebuild counter: " << rebuildCounter << std::endl;
            std::cout << "T : " << std::setw(10) << T << " | ";
            std::cout << "Ek: " << std::setw(10) << Ek << " | ";
            std::cout << "E0: " << std::setw(10) << E0 << " | ";
            std::cout << "E : " << std::setw(10) << E0 + Ek << " | ";
            std::cout << "Nlocal : " << std::setw(10) << atoms.numLocalParticles << " | ";
            std::cout << "Nghost : " << std::setw(10) << atoms.numGhostParticles << std::endl;

            io::dumpCSV("particles_" + std::to_string(i) + ".csv", atoms);
        }
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;

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