#include "Fluctuation.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
TEST(Fluctuation, normalization)
{
    auto histogram0 = data::MultiHistogram("hist", 0_r, 1_r, 10, 1);
    for (auto i = 0; i < 10; ++i)
    {
        histogram0.data(i, 0) = 3;
    }
    auto fluctuation0 = analysis::getFluctuation(histogram0, 2_r, 0);

    auto histogram1 = data::MultiHistogram("hist", 0_r, 2_r, 10, 1);
    for (auto i = 0; i < 10; ++i)
    {
        histogram1.data(i, 0) = 3;
    }
    auto fluctuation1 = analysis::getFluctuation(histogram1, 2_r, 0);

    EXPECT_FLOAT_EQ(fluctuation0, fluctuation1);
}

TEST(Fluctuation, check)
{
    auto histogram = data::MultiHistogram("hist", 0_r, 2_r, 10, 2);
    for (auto i = 0; i < 10; ++i)
    {
        histogram.data(i, 0) = 3;
        histogram.data(i, 1) = 4;
    }

    auto fluctuation0 = analysis::getFluctuation(histogram, 2_r, 0);
    auto fluctuation1 = analysis::getFluctuation(histogram, 2_r, 1);

    EXPECT_FLOAT_EQ(fluctuation0, 0.25_r);
    EXPECT_FLOAT_EQ(fluctuation1, 1_r);
}

TEST(Fluctuation, relation)
{
    auto histogram = data::MultiHistogram("hist", 0_r, 1_r, 10, 2);
    for (auto i = 0; i < 10; ++i)
    {
        histogram.data(i, 0) = i + 1;
        histogram.data(i, 1) = i;
    }

    auto fluctuation0 = analysis::getFluctuation(histogram, 0.5_r, 0);
    auto fluctuation1 = analysis::getFluctuation(histogram, 0.5_r, 1);

    EXPECT_GT(fluctuation0, fluctuation1);
}
}  // namespace analysis
}  // namespace mrmd