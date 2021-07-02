#include "Random.hpp"

#include <gtest/gtest.h>

namespace util
{
TEST(Random, Range)
{
    Random rng;
    for (auto i = 0; i < 10; ++i)
    {
        auto tmp = rng.draw();
        EXPECT_GE(tmp, 0_r);
        EXPECT_LT(tmp, 1_r);
    }
}
}  // namespace util
