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
