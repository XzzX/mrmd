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

#include "angle.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(radToDeg, radToDeg)
{
    EXPECT_FLOAT_EQ(radToDeg(M_PI), 180_r);
    EXPECT_FLOAT_EQ(radToDeg(M_PI * 0.5_r), 90_r);
}

TEST(degToRad, degToRad)
{
    EXPECT_FLOAT_EQ(degToRad(180_r), M_PI);
    EXPECT_FLOAT_EQ(degToRad(90_r), M_PI * 0.5_r);
}

}  // namespace util
}  // namespace mrmd
