#include "communication/AccumulateForce.hpp"

#include <gtest/gtest.h>

#include <data/Atoms.hpp>

namespace mrmd
{
namespace communication
{
namespace impl
{
TEST(AccumulateForceTest, ghostToReal)
{
    // accumulate all force on atom 0
    data::Atoms atoms(101);
    atoms.numLocalAtoms = 1;
    atoms.numGhostAtoms = 100;
    atoms.resize(101);

    IndexView correspondingRealAtom("correspondingRealAtom", 101);
    auto h_correspondingRealAtoms = Kokkos::create_mirror_view(correspondingRealAtom);
    h_correspondingRealAtoms(0) = -1;
    Kokkos::deep_copy(correspondingRealAtom, h_correspondingRealAtoms);

    auto force = atoms.getForce();
    Cabana::deep_copy(force, real_t(1));

    AccumulateForce::ghostToReal(atoms, correspondingRealAtom);

    data::HostAtoms h_atoms(atoms);
    EXPECT_FLOAT_EQ(h_atoms.getForce()(0, 0), real_t(101));
    EXPECT_FLOAT_EQ(h_atoms.getForce()(0, 1), real_t(101));
    EXPECT_FLOAT_EQ(h_atoms.getForce()(0, 2), real_t(101));
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
