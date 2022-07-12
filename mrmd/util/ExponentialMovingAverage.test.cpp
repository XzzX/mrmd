#include "ExponentialMovingAverage.hpp"

#include <gtest/gtest.h>

#include <iostream>

namespace mrmd
{
namespace util
{
TEST(ExponentialMovingAverage, cout)
{
    ExponentialMovingAverage exp(real_t(0.1));
    std::cout << exp << std::endl;
}

TEST(ExponentialMovingAverage, cast)
{
    ExponentialMovingAverage exp(real_t(0.1));
    real_t v = exp + real_t(11);
    EXPECT_FLOAT_EQ(v, real_t(11));
}

TEST(ExponentialMovingAverage, sum)
{
    ExponentialMovingAverage exp(real_t(0.1));
    exp.append(real_t(2));
    EXPECT_FLOAT_EQ(exp, real_t(2));
    exp.append(real_t(3));
    EXPECT_FLOAT_EQ(exp, real_t(2.1));
}

TEST(ExponentialMovingAverage, operator)
{
    ExponentialMovingAverage exp(real_t(0.1));
    exp << real_t(2);
    EXPECT_FLOAT_EQ(exp, real_t(2));
    exp << real_t(3);
    EXPECT_FLOAT_EQ(exp, real_t(2.1));
}

}  // namespace util
}  // namespace mrmd
