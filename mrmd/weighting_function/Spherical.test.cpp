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

#include "Spherical.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace weighting_function
{
TEST(Spherical, monotonous)
{
    std::array<real_t, 3> center = {2_r, 3_r, 4_r};
    real_t atomisticRadius = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Spherical(center, atomisticRadius, hybridRegionDiameter, 2);

    std::array<real_t, 3> pos = center;
    std::array<real_t, 3> delta = {0.1_r, 0.1_r, 0.1_r};
    auto w = weight(pos[0], pos[1], pos[2]);
    for (auto i = 0; i < 60; ++i)
    {
        auto old = w;
        w = weight(pos[0], pos[1], pos[2]);
        EXPECT_LE(w, old);
        pos[0] += delta[0];
        pos[1] += delta[1];
        pos[2] += delta[2];
    }
}

TEST(Spherical, boundaryValues)
{
    std::array<real_t, 3> center = {2_r, 3_r, 4_r};
    real_t atomisticRadius = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Spherical(center, atomisticRadius, hybridRegionDiameter, 2);

    auto w = weight(2.1_r, 3.1_r, 4.1_r);
    EXPECT_FLOAT_EQ(w, 1_r);

    w = weight(6.1_r, 7.1_r, 8.1_r);
    EXPECT_FLOAT_EQ(w, 0_r);
}

TEST(Spherical, derivative)
{
    const auto eps = 1e-10_r;
    std::array<real_t, 3> center = {2_r, 3_r, 4_r};
    real_t atomisticRadius = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Spherical(center, atomisticRadius, hybridRegionDiameter, 2);

    std::array<real_t, 3> pos = center;
    std::array<real_t, 3> delta = {0.1_r, 0.1_r, 0.1_r};
    for (auto i = 0; i < 60; ++i)
    {
        real_t lambda0;
        real_t gradLambdaX0;
        real_t gradLambdaY0;
        real_t gradLambdaZ0;

        real_t lambda1;
        real_t gradLambdaX1;
        real_t gradLambdaY1;
        real_t gradLambdaZ1;

        weight(pos[0], 0_r, 0_r, lambda0, gradLambdaX0, gradLambdaY0, gradLambdaZ0);
        weight(pos[0] + eps, 0_r, 0_r, lambda1, gradLambdaX1, gradLambdaY1, gradLambdaZ1);
        EXPECT_FLOAT_EQ((lambda1 - lambda0) / eps, 0.5_r * (gradLambdaX0 + gradLambdaX1));
        pos[0] += delta[0];
    }
}
}  // namespace weighting_function
}  // namespace mrmd
