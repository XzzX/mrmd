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

#include "communication/UpdateGhostAtoms.hpp"

#include <gtest/gtest.h>

#include <data/Atoms.hpp>

namespace mrmd
{
namespace communication
{
namespace impl
{
struct UpdateGhostAtomsTestData
{
    Vector3D initialDelta;
    Vector3D finalDelta;
};

std::ostream& operator<<(std::ostream& os, const UpdateGhostAtomsTestData& data)
{
    os << "initial delta: [" << data.initialDelta[0] << ", " << data.initialDelta[1] << ", "
       << data.initialDelta[2] << "], ";
    os << "mapped delta: [" << data.finalDelta[0] << ", " << data.finalDelta[1] << ", "
       << data.finalDelta[2] << "]" << std::endl;
    return os;
}

class UpdateGhostAtomsTest : public testing::TestWithParam<UpdateGhostAtomsTestData>
{
protected:
    void SetUp() override
    {
        auto pos = h_atoms.getPos();
        pos(0, 0) = 0.5_r;
        pos(0, 1) = 0.5_r;
        pos(0, 2) = 0.5_r;
        h_atoms.numLocalAtoms = 1;
        h_atoms.numGhostAtoms = 1;
        data::deep_copy(atoms, h_atoms);

        IndexView::host_mirror_type h_correspondingRealAtom("correspondingRealAtom", 2);
        h_correspondingRealAtom(0) = -1;
        h_correspondingRealAtom(1) = 0;
        Kokkos::deep_copy(correspondingRealAtom, h_correspondingRealAtom);
    }

    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, 0.1_r);
    data::HostAtoms h_atoms = data::HostAtoms(2);
    data::Atoms atoms = data::Atoms(2);
    IndexView correspondingRealAtom = IndexView("correspondingRealAtom", 2);
};

TEST_P(UpdateGhostAtomsTest, Check)
{
    auto pos = h_atoms.getPos();
    pos(1, 0) = 0.5_r + GetParam().initialDelta[0];
    pos(1, 1) = 0.5_r + GetParam().initialDelta[1];
    pos(1, 2) = 0.5_r + GetParam().initialDelta[2];
    data::deep_copy(atoms, h_atoms);

    UpdateGhostAtoms::updateOnlyPos(atoms, correspondingRealAtom, subdomain);

    data::deep_copy(h_atoms, atoms);
    pos = h_atoms.getPos();

    EXPECT_FLOAT_EQ(pos(1, 0), 0.5_r + GetParam().finalDelta[0]);
    EXPECT_FLOAT_EQ(pos(1, 1), 0.5_r + GetParam().finalDelta[1]);
    EXPECT_FLOAT_EQ(pos(1, 2), 0.5_r + GetParam().finalDelta[2]);
}

INSTANTIATE_TEST_SUITE_P(
    ShiftX,
    UpdateGhostAtomsTest,
    testing::Values(UpdateGhostAtomsTestData{{0.2_r, 0_r, 0_r}, {1_r, 0_r, 0_r}},
                    UpdateGhostAtomsTestData{{-0.2_r, 0_r, 0_r}, {-1_r, 0_r, 0_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftY,
    UpdateGhostAtomsTest,
    testing::Values(UpdateGhostAtomsTestData{{0_r, 0.2_r, 0_r}, {0_r, 1_r, 0_r}},
                    UpdateGhostAtomsTestData{{0_r, -0.2_r, 0_r}, {0_r, -1_r, 0_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftZ,
    UpdateGhostAtomsTest,
    testing::Values(UpdateGhostAtomsTestData{{0_r, 0_r, 0.2_r}, {0_r, 0_r, 1_r}},
                    UpdateGhostAtomsTestData{{0_r, 0_r, -0.2_r}, {0_r, 0_r, -1_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftXYZ,
    UpdateGhostAtomsTest,
    testing::Values(UpdateGhostAtomsTestData{{0.2_r, 0.2_r, 0.2_r}, {1_r, 1_r, 1_r}},
                    UpdateGhostAtomsTestData{{-0.2_r, -0.2_r, -0.2_r}, {-1_r, -1_r, -1_r}}));

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
