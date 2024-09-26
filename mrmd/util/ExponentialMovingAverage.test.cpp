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

#include "ExponentialMovingAverage.hpp"

#include <gtest/gtest.h>

#include <iostream>

namespace mrmd
{
namespace util
{
TEST(ExponentialMovingAverage, cout)
{
    ExponentialMovingAverage exp(0.1_r);
    std::cout << exp << std::endl;
}

TEST(ExponentialMovingAverage, cast)
{
    ExponentialMovingAverage exp(0.1_r);
    real_t v = exp + 11_r;
    EXPECT_FLOAT_EQ(v, 11_r);
}

TEST(ExponentialMovingAverage, sum)
{
    ExponentialMovingAverage exp(0.1_r);
    exp.append(2_r);
    EXPECT_FLOAT_EQ(exp, 2_r);
    exp.append(3_r);
    EXPECT_FLOAT_EQ(exp, 2.1_r);
}

TEST(ExponentialMovingAverage, operator)
{
    ExponentialMovingAverage exp(0.1_r);
    exp << 2_r;
    EXPECT_FLOAT_EQ(exp, 2_r);
    exp << 3_r;
    EXPECT_FLOAT_EQ(exp, 2.1_r);
}

}  // namespace util
}  // namespace mrmd
