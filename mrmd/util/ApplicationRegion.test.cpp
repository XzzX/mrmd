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

#include "ApplicationRegion.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(ApplicationRegion, testRegion)
{
    Point3D center = {2_r, 3_r, 4_r};
    const auto applicationRegionMin = 2_r; 
    const auto applicationRegionMax = 2_r;
    auto applicationRegion = ApplicationRegion(center, applicationRegionMin, applicationRegionMax);

    EXPECT_TRUE(applicationRegion.isInApplicationRegion(2_r, 10_r, 10_r));
    EXPECT_FALSE(applicationRegion.isInApplicationRegion(3.1_r, 10_r, 10_r));
    EXPECT_FALSE(applicationRegion.isInApplicationRegion(5.2_r, 10_r, 10_r));
}

} // namespace util
} // namespace mrmd