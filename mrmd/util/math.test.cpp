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

#include "math.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(powInt, positiveExponent)
{
    const auto base = -1.5_r;
    EXPECT_FLOAT_EQ(powInt(base, 1), base);
    EXPECT_FLOAT_EQ(powInt(base, 2), base * base);
    EXPECT_FLOAT_EQ(powInt(base, 3), base * base * base);
}

TEST(powInt, zeroExponent)
{
    const auto base = -1.5_r;
    EXPECT_FLOAT_EQ(powInt(base, 0), 1_r);
}

TEST(powInt, negativeExponent)
{
    const auto base = -1.5_r;
    EXPECT_FLOAT_EQ(powInt(base, -1), 1_r / (base));
    EXPECT_FLOAT_EQ(powInt(base, -2), 1_r / (base * base));
    EXPECT_FLOAT_EQ(powInt(base, -3), 1_r / (base * base * base));
}

TEST(approxErfc, STD)
{
    for (auto x = 0.1_r; x < 3_r; x += 0.1_r)
    {
        EXPECT_NEAR(std::erfc(x), approxErfc(x), 1e-6_r);
    }
}
}  // namespace util
}  // namespace mrmd
