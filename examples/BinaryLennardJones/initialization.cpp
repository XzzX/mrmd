#include "initialization.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace mrmd
{
data::Atoms fillDomainWithAtomsSC(const data::Subdomain& subdomain,
                                  const idx_t& numAtoms,
                                  const real_t& fracTypeA,
                                  const real_t& maxVelocity)
{
    assert(fracTypeA < 1_r);
    assert(fracTypeA > 0_r);
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    data::Atoms atoms(numAtoms);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto mass = atoms.getMass();
    auto type = atoms.getType();

    auto numAtomsA = int_c(real_c(numAtoms) * fracTypeA);

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

        mass(idx) = 1_r;

        type(idx) = idx < numAtomsA ? 0 : 1;
        type(idx) = 0;
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

void init(const YAML::Node& config, data::Atoms& atoms, data::Subdomain& subdomain)
{
    subdomain = data::Subdomain({0_r, 0_r, 0_r},
                                {config["box"][0].as<real_t>(),
                                 config["box"][1].as<real_t>(),
                                 config["box"][2].as<real_t>()},
                                0.1_r);
    atoms = fillDomainWithAtomsSC(subdomain,
                                  config["num_atoms"].as<int64_t>(),
                                  config["fraction_type_A"].as<real_t>(),
                                  config["max_velocity"].as<real_t>());
}

}  // namespace mrmd