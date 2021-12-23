#include "MultiHistogram.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace data
{
TEST(MultiHistogram, smoothen_symmetric)
{
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 2);
    histogram.data(5, 0) = 10_r;
    histogram.data(5, 1) = 5_r;

    auto smoothedDensityProfile = smoothen(histogram, 1_r, 3_r);

    for (auto idx = 1; idx < 6; ++idx)
    {
        EXPECT_FLOAT_EQ(smoothedDensityProfile.data(5 - idx, 0),
                        smoothedDensityProfile.data(5 + idx, 0));
        EXPECT_FLOAT_EQ(smoothedDensityProfile.data(5 - idx, 1),
                        smoothedDensityProfile.data(5 + idx, 1));
    }
}

TEST(MultiHistogram, smoothen_constant)
{
    constexpr auto CONST_VAL = 2_r;
    MultiHistogram histogram("histogram", 0_r, 10_r, 11, 1);
    for (auto idx = 0; idx < 11; ++idx)
    {
        histogram.data(idx, 0) = CONST_VAL;
    }

    auto smoothedDensityProfile = smoothen(histogram, 1_r, 3_r);

    for (auto idx = 0; idx < 11; ++idx)
    {
        EXPECT_FLOAT_EQ(smoothedDensityProfile.data(idx, 0), CONST_VAL);
    }
}
}  // namespace data
}  // namespace mrmd