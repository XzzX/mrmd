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
