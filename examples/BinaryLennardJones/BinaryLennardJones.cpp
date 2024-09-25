// Copyright 2024 Sebastian Eibl
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     https://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    MPI_Init(&argc, &argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);

    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    std::string configFilename = "input.yaml";
    CLI::App app{"HAdResS: Binary Lennard Jones"};
    app.add_option("-c,--config", configFilename, "config file");
    CLI11_PARSE(app, argc, argv);

    auto cfg = YAML::LoadFile(configFilename);
    std::cout << "===== CONFIG =====" << std::endl;
    std::cout << cfg << std::endl;
    std::cout << "==================" << std::endl;

    LJ(cfg);
    
    MPI_Finalize();

    return EXIT_SUCCESS;
}