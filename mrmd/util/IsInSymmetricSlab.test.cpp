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

#include "IsInSymmetricSlab.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(IsInSymmetricSlab, testDefaultAxis)
{
    Point3D center = {2_r, 3_r, 4_r};
    const auto slabMin = 2_r;
    const auto slabMax = 4_r;
    auto isInSymmetricSlab = IsInSymmetricSlab(center, slabMin, slabMax);

    EXPECT_TRUE(isInSymmetricSlab(4_r, 10_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(5.7_r, 10_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(6_r, 10_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(-2_r, 10_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(-1.2_r, 10_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(0_r, 10_r, 10_r));

    EXPECT_FALSE(isInSymmetricSlab(3.1_r, 10_r, 10_r));
    EXPECT_FALSE(isInSymmetricSlab(-2.1_r, 10_r, 10_r));
}

TEST(IsInSymmetricSlab, testCustomAxisAndTolerance)
{
    Point3D center = {2_r, 3_r, 4_r};
    const auto slabMin = 2_r;
    const auto slabMax = 4_r;
    const auto tolerance = 0.1_r;
    const auto axis = AXIS::Y;
    auto isInSymmetricSlab = IsInSymmetricSlab(center, slabMin, slabMax, axis, tolerance);

    EXPECT_TRUE(isInSymmetricSlab(10_r, 7_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(10_r, -1_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(10_r, 5.7_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(10_r, 6_r, 10_r));
    EXPECT_TRUE(isInSymmetricSlab(10_r, 0_r, 10_r));

    EXPECT_FALSE(isInSymmetricSlab(10_r, -1.2_r, 10_r));
    EXPECT_FALSE(isInSymmetricSlab(10_r, -2_r, 10_r));
    EXPECT_FALSE(isInSymmetricSlab(10_r, 4_r, 10_r));
    EXPECT_FALSE(isInSymmetricSlab(3.1_r, 10_r, 10_r));
    EXPECT_FALSE(isInSymmetricSlab(-2.1_r, 10_r, 10_r));
}

TEST(IsInSymmetricSlab, testSymmetricInterval)
{
    Point3D center = {2_r, 3_r, 4_r};
    const auto intervalMin = 2_r;
    const auto intervalMax = 4_r;
    auto isInSymmetricInterval = IsInSymmetricSlab(center, intervalMin, intervalMax);

    EXPECT_TRUE(isInSymmetricInterval(4_r));
    EXPECT_TRUE(isInSymmetricInterval(5.7_r));
    EXPECT_TRUE(isInSymmetricInterval(6_r));
    EXPECT_TRUE(isInSymmetricInterval(-2_r));
    EXPECT_TRUE(isInSymmetricInterval(-1.2_r));
    EXPECT_TRUE(isInSymmetricInterval(0_r));

    EXPECT_FALSE(isInSymmetricInterval(3.1_r));
    EXPECT_FALSE(isInSymmetricInterval(-2.1_r));
}

}  // namespace util
}  // namespace mrmd