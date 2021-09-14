#include "LennardJones.hpp"

#include <gtest/gtest.h>

#include <fstream>

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
        potential(idx) = LJ.computeEnergy(x * x, 0);
        force(idx) = -x * LJ.computeForce(x * x, 0);
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();
}

TEST(LennardJones, ExplicitComparison)
{
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3.01_r;
    constexpr real_t rc = 2.5_r * sigma;
    constexpr real_t cappingDistance = 0.1_r;
    const auto cutoffPotential =
        4_r * epsilon * (std::pow(sigma / rc, 12) - std::pow(sigma / rc, 6));

    impl::CappedLennardJonesPotential LJ({cappingDistance}, {rc}, {sigma}, {epsilon}, 1, true);

    constexpr idx_t steps = 100;
    constexpr real_t delta = 0.1_r;
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
            4_r * epsilon * (std::pow(sigma / x, 12) - std::pow(sigma / x, 6)) - cutoffPotential;
        EXPECT_FLOAT_EQ(hPotential(idx), potential);
        auto force = 4_r * epsilon * (-12 * std::pow(sigma / x, 12) + 6 * std::pow(sigma / x, 6)) *
                     x / (x * x);
        EXPECT_FLOAT_EQ(hForce(idx), force);
    }
}

}  // namespace action
}  // namespace mrmd