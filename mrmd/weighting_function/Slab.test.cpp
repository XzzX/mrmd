#include "Slab.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace weighting_function
{
TEST(Slab, testRegion)
{
    std::array<real_t, 3> center = {real_t(2), real_t(3), real_t(4)};
    real_t atomisticRegionDiameter = real_t(2);
    real_t hybridRegionDiameter = real_t(2);
    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);

    EXPECT_TRUE(weight.isInATRegion(real_t(2), real_t(10), real_t(10)));
    EXPECT_FALSE(weight.isInHYRegion(real_t(2), real_t(10), real_t(10)));
    EXPECT_FALSE(weight.isInCGRegion(real_t(2), real_t(10), real_t(10)));

    EXPECT_FALSE(weight.isInATRegion(real_t(3.1), real_t(10), real_t(10)));
    EXPECT_TRUE(weight.isInHYRegion(real_t(3.1), real_t(10), real_t(10)));
    EXPECT_FALSE(weight.isInCGRegion(real_t(3.1), real_t(10), real_t(10)));

    EXPECT_FALSE(weight.isInATRegion(real_t(5.2), real_t(10), real_t(10)));
    EXPECT_FALSE(weight.isInHYRegion(real_t(5.2), real_t(10), real_t(10)));
    EXPECT_TRUE(weight.isInCGRegion(real_t(5.2), real_t(10), real_t(10)));
}

TEST(Slab, monotonous)
{
    std::array<real_t, 3> center = {real_t(2), real_t(3), real_t(4)};
    real_t atomisticRegionDiameter = real_t(2);
    real_t hybridRegionDiameter = real_t(2);
    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);

    std::array<real_t, 3> pos = center;
    std::array<real_t, 3> delta = {real_t(0.1), real_t(0.1), real_t(0.1)};
    real_t tmp;
    real_t w;
    weight(pos[0], pos[1], pos[2], tmp, w, tmp, tmp, tmp);
    for (auto i = 0; i < 60; ++i)
    {
        auto old = w;
        weight(pos[0], pos[1], pos[2], tmp, w, tmp, tmp, tmp);
        EXPECT_LE(w, old);
        pos[0] += delta[0];
        pos[1] += delta[1];
        pos[2] += delta[2];
    }
}

TEST(Slab, boundaryValues)
{
    std::array<real_t, 3> center = {real_t(2), real_t(3), real_t(4)};
    real_t atomisticRegionDiameter = real_t(2);
    real_t hybridRegionDiameter = real_t(2);
    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);

    real_t tmp;
    real_t w;
    weight(real_t(2.9), real_t(3.1), real_t(4.1), tmp, w, tmp, tmp, tmp);
    EXPECT_FLOAT_EQ(w, real_t(1));

    weight(real_t(5.1), real_t(7.1), real_t(8.1), tmp, w, tmp, tmp, tmp);
    EXPECT_FLOAT_EQ(w, real_t(0));
}

// test deactivated since accuracy is to low
// TEST(Slab, derivative)
//{
//    const auto eps = real_t(1e-10);
//    std::array<real_t, 3> center = {real_t(2), real_t(3), real_t(4)};
//    real_t atomisticRegionDiameter = real_t(4);
//    real_t hybridRegionDiameter = real_t(2);
//    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);
//
//    std::array<real_t, 3> pos = center;
//    std::array<real_t, 3> delta = {real_t(0.1), real_t(0.1), real_t(0.1)};
//    for (auto i = 0; i < 60; ++i)
//    {
//        real_t lambda0;
//        real_t lambda1;
//        real_t gradLambda;
//        real_t tmp;  ///< unused dump variable
//
//        weight(pos[0], real_t(0), real_t(0), lambda0, tmp, tmp, tmp);
//        weight(pos[0] + real_t(0.5) * eps, real_t(0), real_t(0), tmp, gradLambda, tmp, tmp);
//        weight(pos[0] + eps, real_t(0), real_t(0), lambda1, tmp, tmp, tmp);
//        EXPECT_FLOAT_EQ((lambda1 - lambda0) / eps, gradLambda);
//        pos[0] += delta[0];
//    }
//}
}  // namespace weighting_function
}  // namespace mrmd
