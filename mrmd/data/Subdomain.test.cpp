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

#include "Subdomain.hpp"

#include <gtest/gtest.h>

#include "datatypes.hpp"

namespace mrmd
{
TEST(Subdomain, Constructor)
{
    data::Subdomain subdomain({1, 2, 3}, {3, 6, 9}, 0.5_r);

    EXPECT_FLOAT_EQ(subdomain.minCorner[0], 1_r);
    EXPECT_FLOAT_EQ(subdomain.minCorner[1], 2_r);
    EXPECT_FLOAT_EQ(subdomain.minCorner[2], 3_r);

    EXPECT_FLOAT_EQ(subdomain.maxCorner[0], 3_r);
    EXPECT_FLOAT_EQ(subdomain.maxCorner[1], 6_r);
    EXPECT_FLOAT_EQ(subdomain.maxCorner[2], 9_r);

    EXPECT_FLOAT_EQ(subdomain.ghostLayerThickness, 0.5_r);

    EXPECT_FLOAT_EQ(subdomain.diameter[0], 2_r);
    EXPECT_FLOAT_EQ(subdomain.diameter[1], 4_r);
    EXPECT_FLOAT_EQ(subdomain.diameter[2], 6_r);

    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[0], 3_r);
    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[1], 5_r);
    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[2], 7_r);

    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[0], 1.5_r);
    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[1], 2.5_r);
    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[2], 3.5_r);

    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[0], 2.5_r);
    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[1], 5.5_r);
    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[2], 8.5_r);

    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[0], 0.5_r);
    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[1], 1.5_r);
    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[2], 2.5_r);

    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[0], 3.5_r);
    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[1], 6.5_r);
    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[2], 9.5_r);
}

TEST(Subdomain, scale)
{
    auto scalingFactor = 0.5_r;
    data::Subdomain subdomain({1, 2, 3}, {3, 6, 9}, 0.2_r);
    subdomain.scale(scalingFactor);

    EXPECT_FLOAT_EQ(subdomain.minCorner[0], 1_r * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.minCorner[1], 2_r * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.minCorner[2], 3_r * scalingFactor);

    EXPECT_FLOAT_EQ(subdomain.maxCorner[0], 3_r * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.maxCorner[1], 6_r * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.maxCorner[2], 9_r * scalingFactor);

    EXPECT_FLOAT_EQ(subdomain.ghostLayerThickness, 0.2_r);

    EXPECT_FLOAT_EQ(subdomain.diameter[0], 2_r * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.diameter[1], 4_r * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.diameter[2], 6_r * scalingFactor);

    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[0], 2_r * scalingFactor + 0.4_r);
    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[1], 4_r * scalingFactor + 0.4_r);
    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[2], 6_r * scalingFactor + 0.4_r);

    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[0], 0.7_r);
    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[1], 1.2_r);
    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[2], 1.7_r);

    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[0], 1.3_r);
    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[1], 2.8_r);
    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[2], 4.3_r);

    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[0], 0.3_r);
    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[1], 0.8_r);
    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[2], 1.3_r);

    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[0], 1.7_r);
    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[1], 3.2_r);
    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[2], 4.7_r);
}
}  // namespace mrmd