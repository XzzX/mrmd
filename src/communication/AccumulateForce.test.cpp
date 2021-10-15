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
    Kokkos::deep_copy(correspondingRealAtom, 0);
    correspondingRealAtom(0) = -1;

    auto force = atoms.getForce();
    Cabana::deep_copy(force, 1_r);

    AccumulateForce::ghostToReal(atoms, correspondingRealAtom);

    EXPECT_FLOAT_EQ(force(0, 0), 101_r);
    EXPECT_FLOAT_EQ(force(0, 1), 101_r);
    EXPECT_FLOAT_EQ(force(0, 2), 101_r);
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
