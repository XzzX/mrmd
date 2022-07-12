#include "ContributeMoleculeForceToAtoms.hpp"

#include <gtest/gtest.h>

#include "test/DiamondFixture.hpp"

namespace mrmd
{
namespace action
{
using ContributeMoleculeForceToAtomsTest = test::DiamondFixture;

TEST_F(ContributeMoleculeForceToAtomsTest, update)
{
    {
        auto hAoSoA =
            Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), molecules.getAoSoA());
        auto force = Cabana::slice<data::Molecules::FORCE>(hAoSoA);
        force(0, 0) = real_t(5);
        force(0, 1) = real_t(6);
        force(0, 2) = real_t(7);
        Cabana::deep_copy(molecules.getAoSoA(), hAoSoA);
    }

    action::ContributeMoleculeForceToAtoms::update(molecules, atoms);

    {
        auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
        auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);
        EXPECT_FLOAT_EQ(force(0, 0), real_t(5) * real_t(0.25));
        EXPECT_FLOAT_EQ(force(0, 1), real_t(6) * real_t(0.25));
        EXPECT_FLOAT_EQ(force(0, 2), real_t(7) * real_t(0.25));
        EXPECT_FLOAT_EQ(force(1, 0), real_t(5) * real_t(0.75));
        EXPECT_FLOAT_EQ(force(1, 1), real_t(6) * real_t(0.75));
        EXPECT_FLOAT_EQ(force(1, 2), real_t(7) * real_t(0.75));
    }
}

}  // namespace action
}  // namespace mrmd