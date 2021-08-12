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
#include "analysis/SystemMomentum.hpp"
#include "analysis/Temperature.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"
#include "io/RestoreTXT.hpp"

using namespace mrmd;

struct Config
{
    bool bOutput = false;

    idx_t nsteps = 2001;
    real_t rc = 2.5;
    real_t skin = 0.3;
    real_t neighborCutoff = rc + skin;
    real_t dt = 0.005;
    real_t temperature = 1_r;
    real_t gamma = 1_r;

    real_t Lx = 33.8585;

    real_t cell_ratio = 0.5_r;

    idx_t estimatedMaxNeighbors = 60;
};

auto fillDomainWithParticlesSC(const data::Subdomain& subdomain,
                               const real_t& spacing,
                               const real_t& maxVelocity)
{
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    auto nx = idx_t(subdomain.diameter[0] / spacing);
    auto ny = idx_t(subdomain.diameter[1] / spacing);
    auto nxny = nx * ny;
    auto nz = idx_t(subdomain.diameter[2] / spacing);
    auto numParticles = nx * ny * nz;
    data::Particles particles(numParticles);

    auto pos = particles.getPos();
    auto vel = particles.getVel();

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t idy, const idx_t idz)
    {
        auto i = idx + idy * nx + idz * nxny;
        pos(i, 0) = real_c(idx) * spacing + subdomain.minCorner[0];
        pos(i, 1) = real_c(idy) * spacing + subdomain.minCorner[1];
        pos(i, 2) = real_c(idz) * spacing + subdomain.minCorner[2];

        auto randGen = RNG.get_state();
        vel(idx, 0) = (randGen.drand() - 0.5_r) * maxVelocity;
        vel(idx, 1) = (randGen.drand() - 0.5_r) * maxVelocity;
        vel(idx, 2) = (randGen.drand() - 0.5_r) * maxVelocity;
        RNG.free_state(randGen);
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
    auto particles = fillDomainWithParticlesSC(subdomain, 1.1_r, 1_r);
    auto rho = particles.numLocalParticles /
               (subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2]);
    std::cout << "rho: " << rho << std::endl;

    communication::GhostLayer ghostLayer(subdomain);
    action::LennardJones LJ(config.rc, 1_r, 1_r);
    action::LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    VerletList verletList;
    Kokkos::Timer timer;
    real_t maxParticleDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    for (auto i = 0; i < config.nsteps; ++i)
    {
        maxParticleDisplacement += action::VelocityVerlet::preForceIntegrate(particles, config.dt);

        if (maxParticleDisplacement >= config.skin * 0.5_r)
        {
            // reset displacement
            maxParticleDisplacement = 0_r;

            ghostLayer.exchangeRealParticles(particles);

            real_t gridDelta[3] = {
                config.neighborCutoff, config.neighborCutoff, config.neighborCutoff};
            LinkedCellList linkedCellList(particles.getPos(),
                                          0,
                                          particles.numLocalParticles,
                                          gridDelta,
                                          subdomain.minCorner.data(),
                                          subdomain.maxCorner.data());
            particles.permute(linkedCellList);

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
        if (config.temperature >= 0)
        {
            langevinThermostat.apply(particles);
        }
        ghostLayer.contributeBackGhostToReal(particles);

        action::VelocityVerlet::postForceIntegrate(particles, config.dt);

        if (config.bOutput && (i % 100 == 0))
        {
            auto E0 = LJ.computeEnergy(particles, verletList);
            auto T = analysis::getTemperature(particles);
            auto systemMomentum = analysis::getSystemMomentum(particles);
            auto Ek = (3_r / 2_r) * real_c(particles.numLocalParticles) * T;
            std::cout << i << ": " << timer.seconds() << std::endl;
            std::cout << "system momentum: " << systemMomentum[0] << " | " << systemMomentum[1]
                      << " | " << systemMomentum[2] << std::endl;
            std::cout << "rebuild counter: " << rebuildCounter << std::endl;
            std::cout << "T : " << std::setw(10) << T << " | ";
            std::cout << "Ek: " << std::setw(10) << Ek << " | ";
            std::cout << "E0: " << std::setw(10) << E0 << " | ";
            std::cout << "E : " << std::setw(10) << E0 + Ek << " | ";
            std::cout << "Nlocal : " << std::setw(10) << particles.numLocalParticles << " | ";
            std::cout << "Nghost : " << std::setw(10) << particles.numGhostParticles << std::endl;
        }
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;

    auto cores = std::getenv("OMP_NUM_THREADS") != nullptr
                     ? std::string(std::getenv("OMP_NUM_THREADS"))
                     : std::string("0");

    std::ofstream fout("ecab.perf", std::ofstream::app);
    fout << cores << ", " << time << ", " << particles.numLocalParticles << ", " << config.nsteps
         << std::endl;
    fout.close();

    // dumpCSV("particles_" + std::to_string(i) + ".csv", particles);

    auto E0 = LJ.computeEnergy(particles, verletList);
    auto T = analysis::getTemperature(particles);

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