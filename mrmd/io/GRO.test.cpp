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

#include <fstream>
#include <string>

#include "DumpGRO.hpp"
#include "RestoreGRO.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "util/simulationSetup.hpp"

namespace mrmd
{
namespace io
{

TEST(GRO, restoreGRO)
{
    data::Subdomain subdomain({0_r, 0_r, 0_r}, {10_r, 10_r, 10_r}, 0.5_r);
    auto atoms = util::fillDomainWithAtoms(subdomain, 1000, 1_r, 1_r);
    data::Atoms restoredAtoms(0);

    dumpGRO("test.gro", atoms, subdomain, 0_r, "test", "RES", {"A"}, false, true);

    restoreGRO("test.gro", subdomain, restoredAtoms);

    EXPECT_EQ(atoms.numLocalAtoms, restoredAtoms.numLocalAtoms);
    EXPECT_EQ(atoms.numGhostAtoms, restoredAtoms.numGhostAtoms);

    auto positions = atoms.getPos();
    auto velocities = atoms.getVel();
    auto restoredPositions = restoredAtoms.getPos();
    auto restoredVelocities = restoredAtoms.getVel();

    for (idx_t atomIdx = 0; atomIdx < atoms.numLocalAtoms; atomIdx++)
    {
        for (idx_t dim = 0; dim < DIMENSIONS; dim++)
        {
            EXPECT_NEAR(positions(atomIdx, dim), restoredPositions(atomIdx, dim), 1e-3_r);
            EXPECT_NEAR(velocities(atomIdx, dim), restoredVelocities(atomIdx, dim), 1e-4_r);
        }
    }
}

}  // namespace io
}  // namespace mrmd
