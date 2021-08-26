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
