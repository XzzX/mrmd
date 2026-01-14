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

#include <gtest/gtest.h>

#include "DumpThermoForce.hpp"
#include "RestoreThermoForce.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace io
{
action::ThermodynamicForce createThermoForce(const idx_t& numBins,
                                             const idx_t& numForces,
                                             const data::Subdomain& subdomain,
                                             const std::vector<real_t>& targetDensities,
                                             const std::vector<real_t>& forceModulations)
{
    real_t binWidth = (subdomain.maxCorner[0] - subdomain.minCorner[0]) / real_c(numBins);
    action::ThermodynamicForce thermodynamicForce(
        targetDensities, subdomain, binWidth, forceModulations);

    MultiView forces("thermoForce", numBins, numForces);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numBins, numForces});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx)
    {
        forces(idx, jdx) = 1_r * (real_c(idx) + 1) * (real_c(jdx) + 1);
    };
    Kokkos::parallel_for("createThermoForce", policy, kernel);
    Kokkos::fence();

    thermodynamicForce.setForce(forces);

    return thermodynamicForce;
}

TEST(ThermoForce, dumpSingleForce)
{
    idx_t numBins = 100;
    idx_t numForces = 1;
    data::Subdomain subdomain({1_r, 2_r, 3_r}, {4_r, 6_r, 8_r}, 0.5_r);
    const std::vector<real_t> targetDensities(numForces, 1_r);
    const std::vector<real_t> forceModulations(numForces, 1_r);

    auto thermodynamicForce1 =
        createThermoForce(numBins, numForces, subdomain, targetDensities, forceModulations);
    dumpThermoForce("dummySingleForce.txt", thermodynamicForce1, 0);

    auto thermodynamicForce2 = restoreThermoForce("dummySingleForce.txt", subdomain);

    ScalarView::HostMirror grid1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), createGrid(thermodynamicForce1.getForce()));
    ScalarView::HostMirror grid2 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), createGrid(thermodynamicForce2.getForce()));
    auto thermoForce1 =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), thermodynamicForce1.getForce(0));
    auto thermoForce2 =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), thermodynamicForce2.getForce(0));

    for (idx_t binNum = 0; binNum < numBins; binNum++)
    {
        EXPECT_FLOAT_EQ(grid1(binNum), grid2(binNum));
        EXPECT_FLOAT_EQ(thermoForce1(binNum), thermoForce2(binNum));
    }
}
TEST(ThermoForce, dumpMultipleForces)
{
    idx_t numBins = 100;
    idx_t numForces = 5;
    data::Subdomain subdomain({1_r, 2_r, 3_r}, {4_r, 6_r, 8_r}, 0.5_r);
    const std::vector<real_t> targetDensities(numForces, 1_r);
    const std::vector<real_t> forceModulations(numForces, 1_r);

    auto thermodynamicForce1 =
        createThermoForce(numBins, numForces, subdomain, targetDensities, forceModulations);

    dumpThermoForce("dummyMultipleForces.txt", thermodynamicForce1);

    auto thermodynamicForce2 =
        restoreThermoForce("dummyMultipleForces.txt", subdomain, targetDensities, forceModulations);

    ScalarView::HostMirror grid1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), createGrid(thermodynamicForce1.getForce()));
    ScalarView::HostMirror grid2 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), createGrid(thermodynamicForce2.getForce()));
    auto thermoForce1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                            thermodynamicForce1.getForce().data);
    auto thermoForce2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                            thermodynamicForce2.getForce().data);

    for (idx_t binNum = 0; binNum < numBins; binNum++)
    {
        EXPECT_FLOAT_EQ(grid1(binNum), grid2(binNum));
        for (idx_t forceNum = 0; forceNum < numForces; forceNum++)
        {
            EXPECT_FLOAT_EQ(thermoForce1(binNum, forceNum), thermoForce2(binNum, forceNum));
        }
    }
}
}  // namespace io
}  // namespace mrmd