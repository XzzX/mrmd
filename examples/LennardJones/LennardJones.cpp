#include "action/LennardJones.hpp"

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

Particles loadParticles(const std::string& filename)
{
    Particles p(100000);
    auto d_AoSoA = p.getAoSoA();
    auto h_AoSoA = Cabana::create_mirror_view(Kokkos::HostSpace(), d_AoSoA);
    auto h_pos = Cabana::slice<Particles::POS>(h_AoSoA);

    std::ifstream fin(filename);

    idx_t idx = 0;
    while (!fin.eof())
    {
        double x, y, z;
        fin >> x >> y >> z;
        if (fin.eof()) break;
        if (std::isnan(x) || std::isnan(y) || std::isnan(z))
        {
            std::cout << "invalid position: " << x << " " << y << " " << z << std::endl;
            exit(EXIT_FAILURE);
        }
        h_pos(idx, 0) = x;
        h_pos(idx, 1) = y;
        h_pos(idx, 2) = z;
        ++idx;
    }

    fin.close();

    Cabana::deep_copy(d_AoSoA, h_AoSoA);
    p.numLocalParticles = idx;
    p.resize(p.numLocalParticles);

    auto vel = p.getVel();
    auto force = p.getForce();
    Cabana::deep_copy(vel, 0_r);
    Cabana::deep_copy(force, 0_r);

    return p;
}

struct Config
{
    bool bOutput = true;

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

void LJ(Config& config)
{
    auto subdomain =
        Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Lx, config.Lx}, config.neighborCutoff);
    auto particles = loadParticles("positions.txt");

    VelocityVerlet integrator(config.dt);
    communication::GhostLayer ghostLayer(subdomain);
    LennardJones LJ(config.rc, 1_r, 1_r);
    LangevinThermostat langevinThermostat(config.gamma, config.temperature, config.dt);
    VerletList verletList;
    Kokkos::Timer timer;
    real_t maxParticleDisplacement = std::numeric_limits<real_t>::max();
    idx_t rebuildCounter = 0;
    for (auto i = 0; i < config.nsteps; ++i)
    {
        maxParticleDisplacement += integrator.preForceIntegrate(particles);

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
        langevinThermostat.applyThermostat(particles);
        ghostLayer.contributeBackGhostToReal(particles);

        integrator.postForceIntegrate(particles);

        if (config.bOutput && (i % 100 == 0))
        {
            auto E0 = LJ.computeEnergy(particles, verletList);
            auto T = analysis::getTemperature(particles);
            auto systemMomentum = analysis::getSystemMomentum(particles);
            auto Ek = (3.0 / 2.0) * particles.numLocalParticles * T;
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

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;
    LJ(config);

    return EXIT_SUCCESS;
}