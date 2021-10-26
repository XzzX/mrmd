#include "SmoothenDensityProfile.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
TEST(SmoothenDensityProfile, symmetric)
{
    data::MultiHistogram histogram("histogram", 0_r, 10_r, 11, 2);
    histogram.data(5, 0) = 10_r;
    histogram.data(5, 1) = 5_r;

    auto smoothedDensityProfile = smoothenDensityProfile(histogram, 1_r, 3_r);

    for (auto idx = 1; idx < 6; ++idx)
    {
        EXPECT_FLOAT_EQ(smoothedDensityProfile.data(5 - idx, 0),
                        smoothedDensityProfile.data(5 + idx, 0));
        EXPECT_FLOAT_EQ(smoothedDensityProfile.data(5 - idx, 1),
                        smoothedDensityProfile.data(5 + idx, 1));
    }
}
}  // namespace analysis
}  // namespace mrmd