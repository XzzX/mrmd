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

#include "CheckRegion.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace weighting_function
{
TEST(CheckRegion, AT)
{
    EXPECT_FALSE(isInATRegion(0_r));
    EXPECT_FALSE(isInATRegion(0.5_r));
    EXPECT_TRUE(isInATRegion(1_r));
}

TEST(CheckRegion, HY)
{
    EXPECT_FALSE(isInHYRegion(0_r));
    EXPECT_TRUE(isInHYRegion(0.5_r));
    EXPECT_FALSE(isInHYRegion(1_r));
}

TEST(CheckRegion, CG)
{
    EXPECT_TRUE(isInCGRegion(0_r));
    EXPECT_FALSE(isInCGRegion(0.5_r));
    EXPECT_FALSE(isInCGRegion(1_r));
}
}  // namespace weighting_function
}  // namespace mrmd
