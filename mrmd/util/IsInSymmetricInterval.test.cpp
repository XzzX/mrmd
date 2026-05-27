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

#include "IsInSymmetricInterval.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(IsInSymmetricInterval, testRegion)
{
    const real_t center = 2_r;
    const auto intervalMin = 2_r;
    const auto intervalMax = 4_r;
    auto isInSymmetricInterval = IsInSymmetricInterval(center, intervalMin, intervalMax);

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