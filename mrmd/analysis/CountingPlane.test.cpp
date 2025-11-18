#include "CountingPlane.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd::analysis
{
template <AXIS axis>
class CountingPlaneTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        assert(atoms.size() >= 1);
        auto pos = atoms.getPos();
        auto vel = atoms.getVel();

        auto policy = Kokkos::RangePolicy<>(0, 10);
        auto kernel = KOKKOS_LAMBDA(idx_t idx)
        {
            pos(idx, 0) = real_c(idx);
            pos(idx, 1) = real_c(idx);
            pos(idx, 2) = real_c(idx);

            vel(idx, 0) = +0_r;
            vel(idx, 1) = +0_r;
            vel(idx, 2) = +0_r;

            if constexpr (axis == AXIS::X)
            {
                if (idx < 5)
                    vel(idx, 0) = +2.5_r;
                else
                    vel(idx, 0) = -2.5_r;
            }
            else if constexpr (axis == AXIS::Y)
            {
                if (idx < 5)
                    vel(idx, 1) = +2.5_r;
                else
                    vel(idx, 1) = -2.5_r;
            }
            else if constexpr (axis == AXIS::Z)
            {
                if (idx < 5)
                    vel(idx, 2) = +2.5_r;
                else
                    vel(idx, 2) = -2.5_r;
            }
        };
        Kokkos::parallel_for(policy, kernel);
        Kokkos::fence();

        atoms.numLocalAtoms = 10;
        atoms.numGhostAtoms = 0;
    }

    // void TearDown() override {}

    data::Atoms atoms = data::Atoms(10);
};

using CountingPlaneTestX = CountingPlaneTest<AXIS::X>;
using CountingPlaneTestY = CountingPlaneTest<AXIS::Y>;
using CountingPlaneTestZ = CountingPlaneTest<AXIS::Z>;

TEST_F(CountingPlaneTestX, ParticlesMovingInXDirection)
{
    // Create plane at x = 5.5, normal pointing in +x direction
    Point3D pointOnPlane = {4.9_r, 0.0_r, 0.0_r};
    Vector3D planeNormal = {1.0_r, 0.0_r, 0.0_r};
    CountingPlane plane(pointOnPlane, planeNormal);

    plane.startCounting(atoms);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();

    Kokkos::parallel_for(
        "MoveParticlesX", Kokkos::RangePolicy<>(0, atoms.numLocalAtoms), KOKKOS_LAMBDA(idx_t idx) {
            pos(idx, 0) += vel(idx, 0);
            pos(idx, 1) += vel(idx, 1);
            pos(idx, 2) += vel(idx, 2);
        });
    Kokkos::fence();

    int64_t count = plane.stopCounting(atoms);

    EXPECT_EQ(count, 5);
}

TEST_F(CountingPlaneTestY, ParticlesMovingInYDirection)
{
    Point3D pointOnPlane = {0.0_r, 5.1_r, 0.0_r};
    Vector3D planeNormal = {0.0_r, -1.0_r, 0.0_r};
    CountingPlane plane(pointOnPlane, planeNormal);

    plane.startCounting(atoms);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();

    Kokkos::parallel_for(
        "MoveParticlesY", Kokkos::RangePolicy<>(0, atoms.numLocalAtoms), KOKKOS_LAMBDA(idx_t idx) {
            pos(idx, 0) += vel(idx, 0);
            pos(idx, 1) += vel(idx, 1);
            pos(idx, 2) += vel(idx, 2);
        });
    Kokkos::fence();

    int64_t count = plane.stopCounting(atoms);

    EXPECT_EQ(count, 4);
}

TEST_F(CountingPlaneTestZ, ParticlesMovingInZDirection)
{
    Point3D pointOnPlane = {0.0_r, 0.0_r, 4.9_r};
    Vector3D planeNormal = {0.0_r, 0.0_r, -1.0_r};
    CountingPlane plane(pointOnPlane, planeNormal);

    plane.startCounting(atoms);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();

    Kokkos::parallel_for(
        "MoveParticlesZ", Kokkos::RangePolicy<>(0, atoms.numLocalAtoms), KOKKOS_LAMBDA(idx_t idx) {
            pos(idx, 0) += vel(idx, 0);
            pos(idx, 1) += vel(idx, 1);
            pos(idx, 2) += vel(idx, 2);
        });
    Kokkos::fence();

    int64_t count = plane.stopCounting(atoms);

    EXPECT_EQ(count, 5);
}

}  // namespace mrmd::analysis
