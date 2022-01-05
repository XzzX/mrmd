#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "DumpH5MDParallel.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace io
{
auto getAtoms()
{
    auto atoms = data::Atoms(10);
    auto pos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, 10);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        pos(idx, 0) = real_c(idx);
        pos(idx, 1) = real_c(idx) + 0.1_r;
        pos(idx, 2) = real_c(idx) + 0.2_r;
    };
    Kokkos::parallel_for("init-atoms", policy, kernel);
    Kokkos::fence();

    atoms.numLocalAtoms = 10;
    atoms.numGhostAtoms = 0;

    return atoms;
}
TEST(H5MD, dump)
{
    auto subdomain = data::Subdomain();
    auto atoms = getAtoms();

    auto mpiInfo = std::make_shared<data::MPIInfo>(MPI_COMM_WORLD);
    auto dump = DumpH5MDParallel(mpiInfo, "XzzX");
    dump.dump("dummy.h5md", subdomain, atoms);
}
}  // namespace io
}  // namespace mrmd