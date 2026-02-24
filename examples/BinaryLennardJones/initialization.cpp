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

#include "initialization.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "data/MPIInfo.hpp"
#include "io/RestoreH5MDParallel.hpp"
#include "util/simulationSetup.hpp"

namespace mrmd
{
void init(const YAML::Node& config, data::Atoms& atoms, data::Subdomain& subdomain)
{
    if (config["restore_file"].IsDefined())
    {
        subdomain = data::Subdomain({0_r, 0_r, 0_r},
                                    {config["box"][0].as<real_t>(),
                                     config["box"][1].as<real_t>(),
                                     config["box"][2].as<real_t>()},
                                    config["ghost_layer_thickness"].as<real_t>());

        auto mpiInfo = std::make_shared<data::MPIInfo>();
        auto io = io::RestoreH5MDParallel(mpiInfo);
        io.restore(config["restore_file"].as<std::string>(), subdomain, atoms);
        return;
    }

    subdomain = data::Subdomain({0_r, 0_r, 0_r},
                                {config["box"][0].as<real_t>(),
                                 config["box"][1].as<real_t>(),
                                 config["box"][2].as<real_t>()},
                                config["ghost_layer_thickness"].as<real_t>());

    atoms = util::fillRectDomainWithTwoAtomisticSpecies(subdomain,
                                                        config["num_atoms"].as<int64_t>(),
                                                        config["fraction_type_A"].as<real_t>(),
                                                        config["max_velocity"].as<real_t>(),
                                                        1_r,
                                                        1_r);
}

data::Molecules initMolecules(const idx_t& numAtoms)
{
    auto molecules = data::Molecules(2 * numAtoms);
    auto offset = molecules.getAtomsOffset();
    auto size = molecules.getNumAtoms();

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        offset(idx) = idx;
        size(idx) = 1;
    };
    Kokkos::parallel_for("initMolecules", policy, kernel);
    Kokkos::fence();

    molecules.numLocalMolecules = numAtoms;

    return molecules;
}

}  // namespace mrmd
