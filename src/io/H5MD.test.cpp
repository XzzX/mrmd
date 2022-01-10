#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "DumpH5MDParallel.hpp"
#include "RestoreH5MDParallel.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace io
{
data::Atoms getAtoms(const std::shared_ptr<data::MPIInfo>& mpiInfo)
{
    auto atoms = data::Atoms(10);
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto type = atoms.getType();
    auto charge = atoms.getCharge();
    auto mass = atoms.getMass();
    auto relativeMass = atoms.getRelativeMass();

    auto policy = Kokkos::RangePolicy<>(0, 10);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        pos(idx, 0) = real_c(idx * mpiInfo->rank);
        pos(idx, 1) = real_c(idx * mpiInfo->rank) + 0.1_r;
        pos(idx, 2) = real_c(idx * mpiInfo->rank) + 0.2_r;

        vel(idx, 0) = real_c(idx * mpiInfo->rank + 1);
        vel(idx, 1) = real_c(idx * mpiInfo->rank + 1) + 0.1_r;
        vel(idx, 2) = real_c(idx * mpiInfo->rank + 1) + 0.2_r;

        force(idx, 0) = real_c(idx * mpiInfo->rank + 2);
        force(idx, 1) = real_c(idx * mpiInfo->rank + 2) + 0.1_r;
        force(idx, 2) = real_c(idx * mpiInfo->rank + 2) + 0.2_r;

        type(idx) = idx + 3;
        charge(idx) = real_c(idx) + 4.1_r;
        mass(idx) = real_c(idx) + 5.2_r;
        relativeMass(idx) = real_c(idx) + 6.3_r;
    };
    Kokkos::parallel_for("init-atoms", policy, kernel);
    Kokkos::fence();

    atoms.numLocalAtoms = 10;
    atoms.numGhostAtoms = 0;

    return atoms;
}
TEST(H5MD, dump)
{
    auto mpiInfo = std::make_shared<data::MPIInfo>(MPI_COMM_WORLD);

    auto subdomain = data::Subdomain();
    auto atoms1 = getAtoms(mpiInfo);

    auto dump = DumpH5MDParallel(mpiInfo, "XzzX");
    dump.dump("dummy.h5md", subdomain, atoms1);

    auto atoms2 = data::Atoms(0);
    auto restore = RestoreH5MDParallel(mpiInfo);
    restore.restore("dummy.h5md", atoms2);

    auto h_atoms1 = data::HostAtoms(atoms1);  // NOLINT
    auto h_atoms2 = data::HostAtoms(atoms2);  // NOLINT
    EXPECT_EQ(h_atoms1.numLocalAtoms, h_atoms2.numLocalAtoms);
    EXPECT_EQ(h_atoms1.numGhostAtoms, h_atoms2.numGhostAtoms);
    for (idx_t idx = 0; idx < h_atoms2.numLocalAtoms; ++idx)
    {
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 0), h_atoms2.getPos()(idx, 0));
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 1), h_atoms2.getPos()(idx, 1));
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 2), h_atoms2.getPos()(idx, 2));

        EXPECT_FLOAT_EQ(h_atoms1.getVel()(idx, 0), h_atoms2.getVel()(idx, 0));
        EXPECT_FLOAT_EQ(h_atoms1.getVel()(idx, 1), h_atoms2.getVel()(idx, 1));
        EXPECT_FLOAT_EQ(h_atoms1.getVel()(idx, 2), h_atoms2.getVel()(idx, 2));

        EXPECT_FLOAT_EQ(h_atoms1.getForce()(idx, 0), h_atoms2.getForce()(idx, 0));
        EXPECT_FLOAT_EQ(h_atoms1.getForce()(idx, 1), h_atoms2.getForce()(idx, 1));
        EXPECT_FLOAT_EQ(h_atoms1.getForce()(idx, 2), h_atoms2.getForce()(idx, 2));

        EXPECT_EQ(h_atoms1.getType()(idx), h_atoms2.getType()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getCharge()(idx), h_atoms2.getCharge()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getMass()(idx), h_atoms2.getMass()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getRelativeMass()(idx), h_atoms2.getRelativeMass()(idx));
    }
}
}  // namespace io
}  // namespace mrmd