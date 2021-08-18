#include "Shake.hpp"

#include <gtest/gtest.h>

#include "test/DiamondFixture.hpp"

namespace mrmd
{
namespace action
{
using ShakeTest = test::DiamondFixture;

TEST_F(ShakeTest, Attraction)
{
    PairView::host_mirror_type bondedPairs("bondedPairs", 1);
    ScalarView::host_mirror_type eqLength("eqLength", 1);

    bondedPairs(0, 0) = 0;
    bondedPairs(0, 1) = 1;
    eqLength(0) = 1_r;

    impl::Shake shake(atoms, 0.1_r);
    shake.setBonds(bondedPairs, eqLength);

    shake.apply(atoms);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Particles::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), -force(1, 0));
    EXPECT_FLOAT_EQ(force(0, 1), -force(1, 1));
    EXPECT_FLOAT_EQ(force(0, 2), -force(1, 2));

    EXPECT_LT(force(0, 0), 0_r);
    EXPECT_GT(force(0, 1), 0_r);
}

TEST_F(ShakeTest, Repulsion)
{
    PairView::host_mirror_type bondedPairs("bondedPairs", 1);
    ScalarView::host_mirror_type eqLength("eqLength", 1);

    bondedPairs(0, 0) = 0;
    bondedPairs(0, 1) = 1;
    eqLength(0) = 2_r;

    impl::Shake shake(atoms, 0.1_r);
    shake.setBonds(bondedPairs, eqLength);

    shake.apply(atoms);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Particles::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), -force(1, 0));
    EXPECT_FLOAT_EQ(force(0, 1), -force(1, 1));
    EXPECT_FLOAT_EQ(force(0, 2), -force(1, 2));

    EXPECT_GT(force(0, 0), 0_r);
    EXPECT_LT(force(0, 1), 0_r);
}

TEST_F(ShakeTest, Molecules)
{
    auto dt = 0.1_r;
    data::BondView::host_mirror_type bonds("bonds", 1);
    bonds(0).idx = 0;
    bonds(0).jdx = 1;
    bonds(0).eqDistance = 1_r;

    MoleculeConstraints mc(1);
    mc.setConstraints(bonds);
    mc.enforceConstraints(molecules, atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Particles::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), -force(1, 0));
    EXPECT_FLOAT_EQ(force(0, 1), -force(1, 1));
    EXPECT_FLOAT_EQ(force(0, 2), -force(1, 2));

    EXPECT_LT(force(0, 0), 0_r);
    EXPECT_GT(force(0, 1), 0_r);
}

TEST_F(ShakeTest, Shrink)
{
    PairView::host_mirror_type bondedPairs("bondedPairs", 4);
    ScalarView::host_mirror_type eqLength("eqLength", 4);

    bondedPairs(0, 0) = 0;
    bondedPairs(0, 1) = 1;
    eqLength(0) = 1_r;

    bondedPairs(1, 0) = 1;
    bondedPairs(1, 1) = 2;
    eqLength(1) = 1_r;

    bondedPairs(2, 0) = 2;
    bondedPairs(2, 1) = 3;
    eqLength(2) = 1_r;

    bondedPairs(3, 0) = 3;
    bondedPairs(3, 1) = 0;
    eqLength(3) = 1_r;

    impl::Shake shake(atoms, 0.1_r);
    shake.setBonds(bondedPairs, eqLength);

    shake.apply(atoms);
    shake.apply(atoms);
    shake.apply(atoms);
    shake.apply(atoms);

    auto newPos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shake.getUpdatedPos());
    auto calcDist = [=](idx_t idx, idx_t jdx)
    {
        auto dx = newPos(idx, 0) - newPos(jdx, 0);
        auto dy = newPos(idx, 1) - newPos(jdx, 1);
        auto dz = newPos(idx, 2) - newPos(jdx, 2);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    EXPECT_FLOAT_EQ(calcDist(0, 1), 1_r);
    EXPECT_FLOAT_EQ(calcDist(1, 2), 1_r);
    EXPECT_FLOAT_EQ(calcDist(2, 3), 1_r);
    EXPECT_FLOAT_EQ(calcDist(3, 0), 1_r);
}

TEST_F(ShakeTest, Grow)
{
    PairView::host_mirror_type bondedPairs("bondedPairs", 4);
    ScalarView::host_mirror_type eqLength("eqLength", 4);

    bondedPairs(0, 0) = 0;
    bondedPairs(0, 1) = 1;
    eqLength(0) = 2_r;

    bondedPairs(1, 0) = 1;
    bondedPairs(1, 1) = 2;
    eqLength(1) = 2_r;

    bondedPairs(2, 0) = 2;
    bondedPairs(2, 1) = 3;
    eqLength(2) = 2_r;

    bondedPairs(3, 0) = 3;
    bondedPairs(3, 1) = 0;
    eqLength(3) = 2_r;

    impl::Shake shake(atoms, 0.1_r);
    shake.setBonds(bondedPairs, eqLength);

    shake.apply(atoms);
    shake.apply(atoms);
    shake.apply(atoms);
    shake.apply(atoms);

    auto newPos = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), shake.getUpdatedPos());
    auto calcDist = [=](idx_t idx, idx_t jdx)
    {
        auto dx = newPos(idx, 0) - newPos(jdx, 0);
        auto dy = newPos(idx, 1) - newPos(jdx, 1);
        auto dz = newPos(idx, 2) - newPos(jdx, 2);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    EXPECT_FLOAT_EQ(calcDist(0, 1), 2_r);
    EXPECT_FLOAT_EQ(calcDist(1, 2), 2_r);
    EXPECT_FLOAT_EQ(calcDist(2, 3), 2_r);
    EXPECT_FLOAT_EQ(calcDist(3, 0), 2_r);
}

}  // namespace action
}  // namespace mrmd