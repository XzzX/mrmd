#include "Shake.hpp"

#include <gtest/gtest.h>

#include "test/DiamondFixture.hpp"

namespace mrmd
{
namespace action
{
using ShakeTest = test::DiamondFixture;

void integratePosition(data::Atoms& atoms, real_t dt)
{
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(idx_t idx)
    {
        auto dtv = dt;
        auto dtfm = 0.5_r * dt * dt / mass(idx);

        pos(idx, 0) = pos(idx, 0) + dtv * vel(idx, 0) + dtfm * force(idx, 0);
        pos(idx, 1) = pos(idx, 1) + dtv * vel(idx, 1) + dtfm * force(idx, 1);
        pos(idx, 2) = pos(idx, 2) + dtv * vel(idx, 2) + dtfm * force(idx, 2);
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();
}

void enforceSingleConstraint(data::Atoms& atoms, real_t dt, real_t eqDistance)
{
    impl::Shake shake(atoms, dt);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<impl::Shake::UnconstraintUpdate>(0, atoms.numLocalAtoms), shake);
    auto policy = Kokkos::RangePolicy<>(0, 1);
    auto kernel = KOKKOS_LAMBDA(idx_t idx) { shake.enforcePositionalConstraint(0, 1, eqDistance); };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();
}

TEST_F(ShakeTest, Attraction)
{
    auto dt = 0.1_r;

    enforceSingleConstraint(atoms, dt, 1_r);
    integratePosition(atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto newPos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), -force(1, 0));
    EXPECT_FLOAT_EQ(force(0, 1), -force(1, 1));
    EXPECT_FLOAT_EQ(force(0, 2), -force(1, 2));

    EXPECT_LT(force(0, 0), 0_r);
    EXPECT_GT(force(0, 1), 0_r);

    auto calcDist = [=](idx_t idx, idx_t jdx)
    {
        auto dx = newPos(idx, 0) - newPos(jdx, 0);
        auto dy = newPos(idx, 1) - newPos(jdx, 1);
        auto dz = newPos(idx, 2) - newPos(jdx, 2);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    EXPECT_FLOAT_EQ(calcDist(0, 1), 1_r);
}

TEST_F(ShakeTest, Repulsion)
{
    auto dt = 0.1_r;

    enforceSingleConstraint(atoms, dt, 2_r);
    integratePosition(atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto newPos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), -force(1, 0));
    EXPECT_FLOAT_EQ(force(0, 1), -force(1, 1));
    EXPECT_FLOAT_EQ(force(0, 2), -force(1, 2));

    EXPECT_GT(force(0, 0), 0_r);
    EXPECT_LT(force(0, 1), 0_r);

    auto calcDist = [=](idx_t idx, idx_t jdx)
    {
        auto dx = newPos(idx, 0) - newPos(jdx, 0);
        auto dy = newPos(idx, 1) - newPos(jdx, 1);
        auto dz = newPos(idx, 2) - newPos(jdx, 2);
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    EXPECT_FLOAT_EQ(calcDist(0, 1), 2_r);
}

void enforceConstraints(data::Atoms& atoms, real_t dt, const data::BondView& bonds)
{
    impl::Shake shake(atoms, dt);
    Kokkos::parallel_for(
        Kokkos::RangePolicy<impl::Shake::UnconstraintUpdate>(0, atoms.numLocalAtoms), shake);
    Kokkos::fence();
    auto policy = Kokkos::RangePolicy<>(0, bonds.extent(0));
    auto kernel = KOKKOS_LAMBDA(idx_t bondIdx)
    {
        shake.enforcePositionalConstraint(
            bonds(bondIdx).idx, bonds(bondIdx).jdx, bonds(bondIdx).eqDistance);
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();
}

TEST_F(ShakeTest, Shrink)
{
    auto force = atoms.getForce();
    Cabana::deep_copy(force, 0_r);
    auto dt = 0.1_r;

    data::BondView::host_mirror_type bonds("bonds", 4);
    bonds(0).idx = 0;
    bonds(0).jdx = 1;
    bonds(0).eqDistance = 1_r;

    bonds(1).idx = 1;
    bonds(1).jdx = 2;
    bonds(1).eqDistance = 1_r;

    bonds(2).idx = 2;
    bonds(2).jdx = 3;
    bonds(2).eqDistance = 1_r;

    bonds(3).idx = 3;
    bonds(3).jdx = 0;
    bonds(3).eqDistance = 1_r;

    for (int iteration = 0; iteration < 10; ++iteration)
    {
        enforceConstraints(
            atoms, dt, Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), bonds));
    }
    integratePosition(atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto newPos = Cabana::slice<data::Atoms::POS>(hAoSoA);
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
    auto force = atoms.getForce();
    auto dt = 0.1_r;

    data::BondView::host_mirror_type bonds("bonds", 4);
    bonds(0).idx = 0;
    bonds(0).jdx = 1;
    bonds(0).eqDistance = 2_r;

    bonds(1).idx = 1;
    bonds(1).jdx = 2;
    bonds(1).eqDistance = 2_r;

    bonds(2).idx = 2;
    bonds(2).jdx = 3;
    bonds(2).eqDistance = 2_r;

    bonds(3).idx = 3;
    bonds(3).jdx = 0;
    bonds(3).eqDistance = 2_r;

    for (int iteration = 0; iteration < 10; ++iteration)
    {
        enforceConstraints(
            atoms, dt, Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), bonds));
    }
    integratePosition(atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto newPos = Cabana::slice<data::Atoms::POS>(hAoSoA);
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

TEST_F(ShakeTest, Molecules)
{
    auto dt = 0.1_r;
    data::BondView::host_mirror_type bonds("bonds", 1);
    bonds(0).idx = 0;
    bonds(0).jdx = 1;
    bonds(0).eqDistance = 1_r;

    MoleculeConstraints mc(2, 1);
    mc.setConstraints(bonds);
    mc.enforcePositionalConstraints(molecules, atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), -force(1, 0));
    EXPECT_FLOAT_EQ(force(0, 1), -force(1, 1));
    EXPECT_FLOAT_EQ(force(0, 2), -force(1, 2));

    EXPECT_LT(force(0, 0), 0_r);
    EXPECT_GT(force(0, 1), 0_r);

    EXPECT_FLOAT_EQ(force(2, 0), -force(3, 0));
    EXPECT_FLOAT_EQ(force(2, 1), -force(3, 1));
    EXPECT_FLOAT_EQ(force(2, 2), -force(3, 2));

    EXPECT_GT(force(2, 0), 0_r);
    EXPECT_LT(force(2, 1), 0_r);
}

}  // namespace action
}  // namespace mrmd