// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    constexpr real_t epsilon = 2_r;
    constexpr real_t sigma = 3.01_r;
    constexpr real_t rc = 2.5_r * sigma;
    constexpr real_t cappingDistance = 0.1_r;
    const auto cutoffPotential =
        4_r * epsilon * (util::powInt(sigma / rc, 12) - util::powInt(sigma / rc, 6));

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
            4_r * epsilon * (util::powInt(sigma / x, 12) - util::powInt(sigma / x, 6)) -
            cutoffPotential;
        EXPECT_FLOAT_EQ(hPotential(idx), potential);
        auto force = 4_r * epsilon *
                     (-12_r * util::powInt(sigma / x, 12) + 6_r * util::powInt(sigma / x, 6)) * x /
                     (x * x);
        EXPECT_FLOAT_EQ(hForce(idx), force);
    }
}

}  // namespace action
}  // namespace mrmd