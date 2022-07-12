#include "CheckRegion.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace weighting_function
{
TEST(CheckRegion, AT)
{
    EXPECT_FALSE(isInATRegion(real_t(0)));
    EXPECT_FALSE(isInATRegion(real_t(0.5)));
    EXPECT_TRUE(isInATRegion(real_t(1)));
}

TEST(CheckRegion, HY)
{
    EXPECT_FALSE(isInHYRegion(real_t(0)));
    EXPECT_TRUE(isInHYRegion(real_t(0.5)));
    EXPECT_FALSE(isInHYRegion(real_t(1)));
}

TEST(CheckRegion, CG)
{
    EXPECT_TRUE(isInCGRegion(real_t(0)));
    EXPECT_FALSE(isInCGRegion(real_t(0.5)));
    EXPECT_FALSE(isInCGRegion(real_t(1)));
}
}  // namespace weighting_function
}  // namespace mrmd
