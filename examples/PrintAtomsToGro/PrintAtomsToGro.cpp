// Copyright 2024 Julian F. Hille
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

#include <Kokkos_Core.hpp>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpGRO.hpp"

using namespace mrmd;

struct Config
{
    static constexpr idx_t numAtoms = 16 * 16 * 16;

    static constexpr real_t sigma = 0.34_r;        ///< units: nm
    static constexpr real_t epsilon = 0.993653_r;  ///< units: kJ/mol
    static constexpr real_t mass = 39.948_r;       ///< units: u

    static constexpr real_t rc = 2.3_r * sigma;
    static constexpr real_t skin = 0.1;
    static constexpr real_t neighborCutoff = rc + skin;

    static constexpr real_t dt = 0.00217;  ///< units: ps

    real_t Lx = 3.196 * 2_r;  ///< units: nm

    real_t cell_ratio = 1.0_r;

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

        vel(idx, 0) = (randGen.drand() - 0.5_r) * maxVelocity;
        vel(idx, 1) = (randGen.drand() - 0.5_r) * maxVelocity;
        vel(idx, 2) = (randGen.drand() - 0.5_r) * maxVelocity;
        RNG.free_state(randGen);

        mass(idx) = Config::mass;
        type(idx) = 0;
        charge(idx) = 0_r;
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

int main(int argc, char* argv[])  // NOLINT
{
    Kokkos::ScopeGuard scope_guard(argc, argv);
    std::cout << "execution space: " << typeid(Kokkos::DefaultExecutionSpace).name() << std::endl;

    Config config;

    auto subdomain = data::Subdomain({0_r, 0_r, 0_r}, {config.Lx, config.Lx, config.Lx}, config.neighborCutoff);
    const auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
    auto atoms = fillDomainWithAtomsSC(subdomain, config.numAtoms, 1_r);

    io::dumpGRO("atoms_initial.gro", atoms, subdomain, 0_r, "Argon", false);

    return EXIT_SUCCESS;
}