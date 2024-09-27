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

#include "PeriodicMapping.hpp"

#include <gtest/gtest.h>

#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
struct TestData
{
    Point3D initialPos;
    Point3D mappedPos;
};

std::ostream& operator<<(std::ostream& os, const TestData& data)
{
    os << "initial pos: [" << data.initialPos[0] << ", " << data.initialPos[1] << ", "
       << data.initialPos[2] << "], ";
    os << "mapped pos: [" << data.mappedPos[0] << ", " << data.mappedPos[1] << ", "
       << data.mappedPos[2] << "]" << std::endl;
    return os;
}

class PeriodicMappingTest : public testing::TestWithParam<TestData>
{
};

TEST_P(PeriodicMappingTest, Check)
{
    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, 0_r);
    data::HostAtoms h_atoms(1);
    auto pos = h_atoms.getPos();
    h_atoms.numLocalAtoms = 1;
    pos(0, 0) = GetParam().initialPos[0];
    pos(0, 1) = GetParam().initialPos[1];
    pos(0, 2) = GetParam().initialPos[2];
    data::Atoms atoms(h_atoms);
    PeriodicMapping::mapIntoDomain(atoms, subdomain);
    data::deep_copy(h_atoms, atoms);
    pos = h_atoms.getPos();
    EXPECT_FLOAT_EQ(pos(0, 0), GetParam().mappedPos[0]);
    EXPECT_FLOAT_EQ(pos(0, 1), GetParam().mappedPos[1]);
    EXPECT_FLOAT_EQ(pos(0, 2), GetParam().mappedPos[2]);
}

INSTANTIATE_TEST_SUITE_P(Inside,
                         PeriodicMappingTest,
                         testing::Values(TestData{{0.4_r, 0.5_r, 0.6_r}, {0.4_r, 0.5_r, 0.6_r}}));

INSTANTIATE_TEST_SUITE_P(MappingX,
                         PeriodicMappingTest,
                         testing::Values(TestData{{1.1_r, 0.5_r, 0.6_r}, {0.1_r, 0.5_r, 0.6_r}},
                                         TestData{{-0.1_r, 0.5_r, 0.6_r}, {0.9_r, 0.5_r, 0.6_r}}));

INSTANTIATE_TEST_SUITE_P(MappingY,
                         PeriodicMappingTest,
                         testing::Values(TestData{{0.4_r, 1.1_r, 0.6_r}, {0.4_r, 0.1_r, 0.6_r}},
                                         TestData{{0.4_r, -0.1_r, 0.6_r}, {0.4_r, 0.9_r, 0.6_r}}));

INSTANTIATE_TEST_SUITE_P(MappingZ,
                         PeriodicMappingTest,
                         testing::Values(TestData{{0.4_r, 0.5_r, 1.1_r}, {0.4_r, 0.5_r, 0.1_r}},
                                         TestData{{0.4_r, 0.5_r, -0.1_r}, {0.4_r, 0.5_r, 0.9_r}}));

INSTANTIATE_TEST_SUITE_P(MappingXYZ,
                         PeriodicMappingTest,
                         testing::Values(TestData{{1.1_r, 1.2_r, 1.3_r}, {0.1_r, 0.2_r, 0.3_r}},
                                         TestData{{-0.3_r, -0.2_r, -0.1_r},
                                                  {0.7_r, 0.8_r, 0.9_r}}));

}  // namespace impl
}  // namespace communication
}  // namespace mrmd