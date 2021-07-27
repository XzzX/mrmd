#include "SmoothenDensityProfile.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
TEST(SmoothenDensityProfile, symmetric)
{
    data::Histogram histogram("histogram", 0_r, 10_r, 11);
    histogram.data(5) = 10_r;

    auto smoothedDensityProfile = smoothenDensityProfile(histogram, 1_r, 3_r);

    for (auto idx = 1; idx < 6; ++idx)
    {
        EXPECT_FLOAT_EQ(smoothedDensityProfile.data(5 - idx), smoothedDensityProfile.data(5 + idx));
    }
}
}  // namespace analysis
}  // namespace mrmd