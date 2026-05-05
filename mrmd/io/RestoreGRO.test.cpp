// Copyright 2024 Sebastian Eibl
// Copyright 2026 Julian Friedrich Hille
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

#include "DumpGRO.hpp"
#include "RestoreGRO.hpp"
#include "data/Subdomain.hpp"
#include "util/simulationSetup.hpp"

namespace mrmd
{
namespace io
{
TEST(RestoreGRO, restoreGRO)
{
    data::Subdomain subdomain({0_r, 0_r, 0_r}, {10_r, 10_r, 10_r}, 0.5_r);
    auto atoms = util::fillDomainWithAtoms(subdomain, 1000, 1_r, 1_r);

    dumpGRO("test.gro", atoms, subdomain, 0_r, "test", "RES", {"A"}, false, true);

//    auto restoredAtoms = restoreGRO("test.gro", subdomain);
//
//    EXPECT_EQ(atoms.numLocalAtoms, restoredAtoms.numLocalAtoms);
//    EXPECT_EQ(atoms.numGhostAtoms, restoredAtoms.numGhostAtoms);
//
//    auto positions = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getPos());
//    auto velocities = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getVel());
//    auto restoredPositions =
//        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), restoredAtoms.getPos());
//    auto restoredVelocities =
//        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), restoredAtoms.getVel());
//
//    for (idx_t atomNum = 0; atomNum < atoms.numLocalAtoms; atomNum++)
//    {
//        for (idx_t dim = 0; dim < DIMENSIONS; dim++)
//        {
//            EXPECT_FLOAT_EQ(positions(atomNum, dim), restoredPositions(atomNum, dim));
//            EXPECT_FLOAT_EQ(velocities(atomNum, dim), restoredVelocities(atomNum, dim));
//        }
//    }
}
}  // namespace io
}  // namespace mrmd