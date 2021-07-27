#include "SmoothenDensityProfile.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
TEST(SmoothenDensityProfile, symmetric)
{
    ScalarView histogram("histogram", 11);
    histogram(5) = 10_r;

    auto smoothedDensityProfile = smoothenDensityProfile(histogram, 1_r, 1_r, 3_r);

    for (auto idx = 1; idx < 6; ++idx)
    {
        EXPECT_FLOAT_EQ(smoothedDensityProfile(5 - idx), smoothedDensityProfile(5 + idx));
    }
}
}  // namespace analysis
}  // namespace mrmd