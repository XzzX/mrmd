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
#include "NVT.hpp"
#include "SPARTIAN.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "initialization.hpp"

using namespace mrmd;

void LJ(YAML::Node& config)
{
    data::Subdomain subdomain;
    data::Atoms atoms(0);

    auto initializationConfig = config["initialization"];
    init(initializationConfig, atoms, subdomain);

    auto NPTConfig = config["NPT"];
    npt(NPTConfig, atoms, subdomain);

    auto NVTConfig = config["NVT"];
    nvt(NVTConfig, atoms, subdomain);

    atoms.numGhostAtoms = 0;
    auto molecules = initMolecules(atoms.numLocalAtoms);

    auto spartianConfig = config["SPARTIAN"];
    spartian(spartianConfig, molecules, atoms, subdomain);
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    auto cfg = YAML::LoadFile("input.yaml");
    LJ(cfg);

    return EXIT_SUCCESS;
}