#include "angle.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace util
{
TEST(radToDeg, radToDeg)
{
    EXPECT_FLOAT_EQ(radToDeg(M_PI), real_t(180));
    EXPECT_FLOAT_EQ(radToDeg(M_PI * real_t(0.5)), real_t(90));
}

TEST(degToRad, degToRad)
{
    EXPECT_FLOAT_EQ(degToRad(real_t(180)), M_PI);
    EXPECT_FLOAT_EQ(degToRad(real_t(90)), M_PI * real_t(0.5));
}

}  // namespace util
}  // namespace mrmd
