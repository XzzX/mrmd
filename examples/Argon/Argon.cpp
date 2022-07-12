#include <fmt/format.h>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
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
#include "io/DumpCSV.hpp"
#include "io/DumpGRO.hpp"
#include "io/RestoreTXT.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = true;
    idx_t outputInterval = -1;

    idx_t nsteps = 400001;
    static constexpr idx_t numAtoms = 16 * 16 * 16;

    static constexpr real_t sigma = real_t(0.34);        ///< units: nm
    static constexpr real_t epsilon = real_t(0.993653);  ///< units: kJ/mol
    static constexpr real_t mass = real_t(39.948);       ///< units: u

    static constexpr real_t rc = real_t(2.3) * sigma;
    static constexpr real_t skin = 0.1;
    static constexpr real_t neighborCutoff = rc + skin;

    static constexpr real_t dt = 0.00217;  ///< units: ps
    real_t temperature = real_t(3.0);
    real_t gamma = real_t(0.7) / dt;

    real_t Lx = 3.196 * real_t(2);  ///< units: nm

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
    auto type = atoms.getType();
    auto charge = atoms.getCharge();

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

        mass(idx) = Config::mass;
        type(idx) = 0;
        charge(idx) = real_t(0);
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
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numAtoms, real_t(1));
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    io::dumpGRO("atoms_initial.gro", atoms, subdomain, real_t(0), "Argon", false);

    communication::GhostLayer ghostLayer;
    action::LennardJones LJ(config.rc, config.sigma, config.epsilon, real_t(0.7) * config.sigma);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    analysis::MeanSquareDisplacement meanSquareDisplacement;
    meanSquareDisplacement.reset(atoms);
    auto msd = real_t(0);
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

        if (maxAtomDisplacement >= config.skin * real_t(0.5))
        {
            // reset displacement
            maxAtomDisplacement = real_t(0);

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            //            real_t gridDelta[3] = {
            //                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            //            LinkedCellList linkedCellList(atoms.getPos(),
            //                                          0,
            //                                          atoms.numLocalAtoms,
            //                                          gridDelta,
            //                                          subdomain.minCorner.data(),
            //                                          subdomain.maxCorner.data());
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

        ghostLayer.contributeBackGhostToReal(atoms);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto E0 = LJ.getEnergy() / real_c(atoms.numLocalAtoms);
            auto Ek = analysis::getMeanKineticEnergy(atoms);
            auto systemMomentum = analysis::getSystemMomentum(atoms);
            auto T = (real_t(2) / real_t(3)) * Ek;
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

            io::dumpGRO(fmt::format("argon_{:0>6}.gro", step),
                        atoms,
                        subdomain,
                        step * config.dt,
                        "Argon",
                        false);

            //            io::dumpCSV("atoms_" + std::to_string(step) + ".csv", atoms,
            //            false);
        }

        if (step % 1000 == 0)
        {
            msd = meanSquareDisplacement.calc(atoms, subdomain) / (real_t(1000) * config.dt);
            if ((config.temperature > real_t(0)) && (step > 5000))
            {
                config.temperature -= real_t(7.8e-3);
                if (config.temperature < real_t(0)) config.temperature = real_t(0);
            }

            langevinThermostat.set(config.gamma, config.temperature * real_t(0.5), config.dt);
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