#include "math.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(powInt, positiveExponent)
{
    const auto base = real_t(-1.5);
    EXPECT_FLOAT_EQ(powInt(base, 1), base);
    EXPECT_FLOAT_EQ(powInt(base, 2), base * base);
    EXPECT_FLOAT_EQ(powInt(base, 3), base * base * base);
}

TEST(powInt, zeroExponent)
{
    const auto base = real_t(-1.5);
    EXPECT_FLOAT_EQ(powInt(base, 0), real_t(1));
}

TEST(powInt, negativeExponent)
{
    const auto base = real_t(-1.5);
    EXPECT_FLOAT_EQ(powInt(base, -1), real_t(1) / (base));
    EXPECT_FLOAT_EQ(powInt(base, -2), real_t(1) / (base * base));
    EXPECT_FLOAT_EQ(powInt(base, -3), real_t(1) / (base * base * base));
}

TEST(approxErfc, STD)
{
    for (auto x = real_t(0.1); x < real_t(3); x += real_t(0.1))
    {
        EXPECT_NEAR(std::erfc(x), approxErfc(x), real_t(1e-6));
    }
}
}  // namespace util
}  // namespace mrmd
