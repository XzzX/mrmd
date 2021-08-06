#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "action/VelocityVerlet.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"

using namespace mrmd;

struct Config
{
    idx_t nsteps = 2001;
    real_t dt = 0.005;
};

data::Particles initParticles()
{
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);
    auto particles = data::Particles(1000000);
    auto pos = particles.getPos();
    auto vel = particles.getVel();
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {100, 100, 100});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t idy, const idx_t idz)
    {
        auto i = idx + idy * 100 + idz * 10000;
        pos(i, 0) = real_c(idx) + 0.5_r;
        pos(i, 1) = real_c(idy) + 0.5_r;
        pos(i, 2) = real_c(idz) + 0.5_r;

        auto randGen = RNG.get_state();

        vel(idx, 0) = (randGen.drand() - 0.5_r) * 2_r;
        vel(idx, 1) = (randGen.drand() - 0.5_r) * 2_r;
        vel(idx, 2) = (randGen.drand() - 0.5_r) * 2_r;

        // Give the state back, which will allow another thread to acquire it
        RNG.free_state(randGen);
    };
    Kokkos::parallel_for("initParticles", policy, kernel);
    particles.numLocalParticles = 1000000;
    particles.numGhostParticles = 0;
    return particles;
}

void IdealGas(const Config& config)
{
    auto subdomain = data::Subdomain({0_r, 0_r, 0_r}, {100_r, 100_r, 100_r}, 1_r);
    auto particles = initParticles();

    communication::GhostLayer ghostLayer(subdomain);
    Kokkos::Timer timer;
    for (auto i = 0; i < config.nsteps; ++i)
    {
        action::VelocityVerlet::preForceIntegrate(particles, config.dt);

        ghostLayer.exchangeRealParticles(particles);

        if (i % 100 == 0)
        {
            io::dumpCSV("particles_" + std::to_string(i) + ".csv", particles);
        }

        action::VelocityVerlet::postForceIntegrate(particles, config.dt);
    }
    auto time = timer.seconds();
    std::cout << time << std::endl;
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    CLI::App app{"ideal gas benchmark application"};
    app.add_option("-n,--nsteps", config.nsteps, "number of simulation steps");
    CLI11_PARSE(app, argc, argv);

    IdealGas(config);

    return EXIT_SUCCESS;
}