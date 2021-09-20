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
#include "action/LimitAcceleration.hpp"
#include "action/LimitVelocity.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Particles.hpp"
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

    idx_t nsteps = 2001;
    static constexpr idx_t numParticles = 16 * 16 * 16;

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
};

data::Particles fillDomainWithParticlesSC(const data::Subdomain& subdomain,
                                          const idx_t& numParticles,
                                          const real_t& maxVelocity)
{
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    data::Particles particles(numParticles);

    auto pos = particles.getPos();
    auto vel = particles.getVel();
    auto mass = particles.getMass();
    auto type = particles.getType();
    auto charge = particles.getCharge();

    auto policy = Kokkos::RangePolicy<>(0, numParticles);
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
    };
    Kokkos::parallel_for("fillDomainWithParticlesSC", policy, kernel);

    particles.numLocalParticles = numParticles;
    particles.numGhostParticles = 0;
    return particles;
}

void LJ(Config& config)
{
    auto subdomain =
        data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Lx, config.Lx}, config.neighborCutoff);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto particles = fillDomainWithParticlesSC(subdomain, config.numParticles, 1_r);
    auto rho = real_c(particles.numLocalParticles) / volume;
    std::cout << "rho: " << rho << std::endl;

    io::dumpGRO("particles_initial.gro", particles, subdomain, 0_r, "Argon", false);

    communication::GhostLayer ghostLayer(subdomain);
    action::LennardJones LJ(config.rc, config.sigma, config.epsilon, 0.7_r * config.sigma);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    VerletList verletList;
    Kokkos::Timer timer;
    real_t maxParticleDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    util::printTable("step", "time", "T", "Ek", "E0", "E", "Nlocal", "Nghost");
    util::printTableSep("step", "time", "T", "Ek", "E0", "E", "Nlocal", "Nghost");

    std::ofstream fStat("statistics.txt");
    for (auto step = 0; step < config.nsteps; ++step)
    {
        maxParticleDisplacement += action::VelocityVerlet::preForceIntegrate(particles, config.dt);

        if (maxParticleDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxParticleDisplacement = 0_r;

            ghostLayer.exchangeRealParticles(particles);

            //            real_t gridDelta[3] = {
            //                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            //            LinkedCellList linkedCellList(particles.getPos(),
            //                                          0,
            //                                          particles.numLocalParticles,
            //                                          gridDelta,
            //                                          subdomain.minCorner.data(),
            //                                          subdomain.maxCorner.data());
            //            particles.permute(linkedCellList);

            ghostLayer.createGhostParticles(particles);
            verletList.build(particles.getPos(),
                             0,
                             particles.numLocalParticles,
                             config.neighborCutoff,
                             config.cell_ratio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data(),
                             config.estimatedMaxNeighbors);
            ++rebuildCounter;
        }
        else
        {
            ghostLayer.updateGhostParticles(particles);
        }

        auto force = particles.getForce();
        Cabana::deep_copy(force, 0_r);

        LJ.applyForces(particles, verletList);

        if (step % 100 == 0)
        {
            if ((config.temperature > 0_r) && (step > 5000))
            {
                config.temperature -= 7.8e-4_r;
                if (config.temperature < 0_r) config.temperature = 0_r;
            }
            langevinThermostat.set(config.gamma, config.temperature, config.dt);
            langevinThermostat.apply(particles);
        }

        ghostLayer.contributeBackGhostToReal(particles);

        if (step < 5000)
        {
            //            action::limitAccelerationPerComponent(particles, 10_r);
            //            action::limitVelocityPerComponent(particles, 1_r);
        }

        action::VelocityVerlet::postForceIntegrate(particles, config.dt);

        if (config.bOutput && (step % config.outputInterval == 0))
        {
            auto E0 = LJ.computeEnergy(particles, verletList);
            auto Ek = analysis::getKineticEnergy(particles);
            auto systemMomentum = analysis::getSystemMomentum(particles);
            auto T = (2_r / (3_r * real_c(particles.numLocalParticles))) * Ek;
            //            std::cout << "system momentum: " << systemMomentum[0] << " | " <<
            //            systemMomentum[1]
            //                      << " | " << systemMomentum[2] << std::endl;
            //            std::cout << "rebuild counter: " << rebuildCounter << std::endl;
            util::printTable(step,
                             timer.seconds(),
                             T,
                             Ek,
                             E0,
                             E0 + Ek,
                             particles.numLocalParticles,
                             particles.numGhostParticles);

            fStat << step << " " << timer.seconds() << " " << T << " " << Ek << " " << E0 << " "
                  << E0 + Ek << " " << particles.numLocalParticles << " "
                  << particles.numGhostParticles << " " << std::endl;

            io::dumpGRO("particles_" + std::to_string(step) + ".gro",
                        particles,
                        subdomain,
                        step * config.dt,
                        "Argon",
                        false);

            //            io::dumpCSV("particles_" + std::to_string(step) + ".csv", particles,
            //            false);
        }
    }
    fStat.close();
    auto time = timer.seconds();
    std::cout << time << std::endl;

    auto cores = util::getEnvironmentVariable("OMP_NUM_THREADS");

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << particles.numLocalParticles << ", " << config.nsteps
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