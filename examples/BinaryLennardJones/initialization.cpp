#include "initialization.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "data/MPIInfo.hpp"
#include "io/RestoreH5MDParallel.hpp"

namespace mrmd
{
data::Atoms fillDomainWithAtomsSC(const data::Subdomain& subdomain,
                                  const idx_t& numAtoms,
                                  const real_t& fracTypeA,
                                  const real_t& maxVelocity)
{
    assert(fracTypeA < real_t(1));
    assert(fracTypeA > real_t(0));
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    data::Atoms atoms(numAtoms);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto mass = atoms.getMass();
    auto relativeMass = atoms.getRelativeMass();
    auto type = atoms.getType();

    auto numAtomsA = int_c(real_c(numAtoms) * fracTypeA);

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto randGen = RNG.get_state();
        pos(idx, 0) = randGen.drand() * subdomain.diameter[0] + subdomain.minCorner[0];
        pos(idx, 1) = randGen.drand() * subdomain.diameter[1] + subdomain.minCorner[1];
        pos(idx, 2) = randGen.drand() * subdomain.diameter[2] + subdomain.minCorner[2];

        vel(idx, 0) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        vel(idx, 1) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        vel(idx, 2) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        RNG.free_state(randGen);

        mass(idx) = real_t(1);
        relativeMass(idx) = real_t(1);

        type(idx) = idx < numAtomsA ? 0 : 1;
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

void init(const YAML::Node& config, data::Atoms& atoms, data::Subdomain& subdomain)
{
    if (config["restore_file"].IsDefined())
    {
        subdomain = data::Subdomain({real_t(0), real_t(0), real_t(0)},
                                    {config["box"][0].as<real_t>(),
                                     config["box"][1].as<real_t>(),
                                     config["box"][2].as<real_t>()},
                                    config["ghost_layer_thickness"].as<real_t>());

        auto mpiInfo = std::make_shared<data::MPIInfo>(MPI_COMM_WORLD);
        auto io = io::RestoreH5MDParallel(mpiInfo);
        io.restore(config["restore_file"].as<std::string>(), subdomain, atoms);
        return;
    }

    subdomain = data::Subdomain({real_t(0), real_t(0), real_t(0)},
                                {config["box"][0].as<real_t>(),
                                 config["box"][1].as<real_t>(),
                                 config["box"][2].as<real_t>()},
                                config["ghost_layer_thickness"].as<real_t>());

    atoms = fillDomainWithAtomsSC(subdomain,
                                  config["num_atoms"].as<int64_t>(),
                                  config["fraction_type_A"].as<real_t>(),
                                  config["max_velocity"].as<real_t>());
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