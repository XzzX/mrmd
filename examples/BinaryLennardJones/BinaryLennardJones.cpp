#include <fmt/format.h>
#include <yaml-cpp/yaml.h>

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "NPT.hpp"
#include "action/BerendsenThermostat.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "initialization.hpp"
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

void LJ(YAML::Node& config)
{
    auto initializationConfig = config["initialization"];
    data::Subdomain subdomain;
    data::Atoms atoms(0);
    init(initializationConfig, atoms, subdomain);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    std::cout << "rho: " << rho << std::endl;

    auto NPTConfig = config["NPT"];
    npt(NPTConfig, atoms, subdomain);
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    auto cfg = YAML::LoadFile("input.yaml");
    LJ(cfg);

    return EXIT_SUCCESS;
}