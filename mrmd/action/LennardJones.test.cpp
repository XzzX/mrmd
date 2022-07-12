#include "LennardJones.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "util/math.hpp"

namespace mrmd
{
namespace action
{
void calcPotentialAndForce(impl::CappedLennardJonesPotential& LJ,
                           ScalarView& force,
                           ScalarView& potential,
                           idx_t steps,
                           real_t startingPos,
                           real_t delta)
{
    auto policy = Kokkos::RangePolicy<>(0, steps);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto x = startingPos + real_c(idx) * delta;
        auto forceAndEnergy = LJ.computeForceAndEnergy(x * x, 0);
        potential(idx) = forceAndEnergy.energy;
        force(idx) = -x * forceAndEnergy.forceFactor;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();
}

TEST(LennardJones, ExplicitComparison)
{
    constexpr real_t epsilon = real_t(2);
    constexpr real_t sigma = real_t(3.01);
    constexpr real_t rc = real_t(2.5) * sigma;
    constexpr real_t cappingDistance = real_t(0.1);
    const auto cutoffPotential =
        real_t(4) * epsilon * (util::powInt(sigma / rc, 12) - util::powInt(sigma / rc, 6));

    impl::CappedLennardJonesPotential LJ({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, true);

    constexpr idx_t steps = 100;
    constexpr real_t delta = real_t(0.1);
    constexpr real_t startingPos = cappingDistance + delta;
    EXPECT_GT(startingPos + real_c(steps) * delta, rc);

    ScalarView force("force", steps);
    ScalarView potential("potential", steps);

    calcPotentialAndForce(LJ, force, potential, steps, startingPos, delta);

    auto hForce = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), force);
    auto hPotential = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), potential);
    for (idx_t idx = 0; idx < steps; ++idx)
    {
        auto x = startingPos + real_c(idx) * delta;
        auto potential =
            real_t(4) * epsilon * (util::powInt(sigma / x, 12) - util::powInt(sigma / x, 6)) -
            cutoffPotential;
        EXPECT_FLOAT_EQ(hPotential(idx), potential);
        auto force =
            real_t(4) * epsilon *
            (real_t(-12) * util::powInt(sigma / x, 12) + real_t(6) * util::powInt(sigma / x, 6)) *
            x / (x * x);
        EXPECT_FLOAT_EQ(hForce(idx), force);
    }
}

}  // namespace action
}  // namespace mrmd