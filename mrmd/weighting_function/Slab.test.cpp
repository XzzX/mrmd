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

#include "Slab.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace weighting_function
{
TEST(Slab, testRegion)
{
    Point3D center = {2_r, 3_r, 4_r};
    real_t atomisticRegionDiameter = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);

    EXPECT_TRUE(weight.isInATRegion(2_r, 10_r, 10_r));
    EXPECT_FALSE(weight.isInHYRegion(2_r, 10_r, 10_r));
    EXPECT_FALSE(weight.isInCGRegion(2_r, 10_r, 10_r));

    EXPECT_FALSE(weight.isInATRegion(3.1_r, 10_r, 10_r));
    EXPECT_TRUE(weight.isInHYRegion(3.1_r, 10_r, 10_r));
    EXPECT_FALSE(weight.isInCGRegion(3.1_r, 10_r, 10_r));

    EXPECT_FALSE(weight.isInATRegion(5.2_r, 10_r, 10_r));
    EXPECT_FALSE(weight.isInHYRegion(5.2_r, 10_r, 10_r));
    EXPECT_TRUE(weight.isInCGRegion(5.2_r, 10_r, 10_r));
}

TEST(Slab, monotonous)
{
    Point3D center = {2_r, 3_r, 4_r};
    real_t atomisticRegionDiameter = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);

    Point3D pos = center;
    Vector3D delta = {0.1_r, 0.1_r, 0.1_r};
    real_t tmp;
    real_t w;
    weight(pos[0], pos[1], pos[2], tmp, w, tmp, tmp, tmp);
    for (auto i = 0; i < 60; ++i)
    {
        auto old = w;
        weight(pos[0], pos[1], pos[2], tmp, w, tmp, tmp, tmp);
        EXPECT_LE(w, old);
        pos[0] += delta[0];
        pos[1] += delta[1];
        pos[2] += delta[2];
    }
}

TEST(Slab, boundaryValues)
{
    Point3D center = {2_r, 3_r, 4_r};
    real_t atomisticRegionDiameter = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);

    real_t tmp;
    real_t w;
    weight(2.9_r, 3.1_r, 4.1_r, tmp, w, tmp, tmp, tmp);
    EXPECT_FLOAT_EQ(w, 1_r);

    weight(5.1_r, 7.1_r, 8.1_r, tmp, w, tmp, tmp, tmp);
    EXPECT_FLOAT_EQ(w, 0_r);
}

// test deactivated since accuracy is to low
// TEST(Slab, derivative)
//{
//    const auto eps = 1e-10_r;
//    Point3D center = {2_r, 3_r, 4_r};
//    real_t atomisticRegionDiameter = 4_r;
//    real_t hybridRegionDiameter = 2_r;
//    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);
//
//    Point3D pos = center;
//    Vector3D delta = {0.1_r, 0.1_r, 0.1_r};
//    for (auto i = 0; i < 60; ++i)
//    {
//        real_t lambda0;
//        real_t lambda1;
//        real_t gradLambda;
//        real_t tmp;  ///< unused dump variable
//
//        weight(pos[0], 0_r, 0_r, lambda0, tmp, tmp, tmp);
//        weight(pos[0] + 0.5_r * eps, 0_r, 0_r, tmp, gradLambda, tmp, tmp);
//        weight(pos[0] + eps, 0_r, 0_r, lambda1, tmp, tmp, tmp);
//        EXPECT_FLOAT_EQ((lambda1 - lambda0) / eps, gradLambda);
//        pos[0] += delta[0];
//    }
//}
}  // namespace weighting_function
}  // namespace mrmd
