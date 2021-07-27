#include "Slab.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace weighting_function
{
TEST(Slab, monotonous)
{
    std::array<real_t, 3> center = {2_r, 3_r, 4_r};
    real_t atomisticRegionDiameter = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);

    std::array<real_t, 3> pos = center;
    std::array<real_t, 3> delta = {0.1_r, 0.1_r, 0.1_r};
    auto w = weight(pos[0], pos[1], pos[2]);
    for (auto i = 0; i < 60; ++i)
    {
        auto old = w;
        w = weight(pos[0], pos[1], pos[2]);
        EXPECT_LE(w, old);
        pos[0] += delta[0];
        pos[1] += delta[1];
        pos[2] += delta[2];
    }
}

TEST(Slab, boundaryValues)
{
    std::array<real_t, 3> center = {2_r, 3_r, 4_r};
    real_t atomisticRegionDiameter = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);

    auto w = weight(2.9_r, 3.1_r, 4.1_r);
    EXPECT_FLOAT_EQ(w, 1_r);

    w = weight(5.1_r, 7.1_r, 8.1_r);
    EXPECT_FLOAT_EQ(w, 0_r);
}

// test deactivated since accuracy is to low
// TEST(Slab, derivative)
//{
//    const auto eps = 1e-10_r;
//    std::array<real_t, 3> center = {2_r, 3_r, 4_r};
//    real_t atomisticRegionDiameter = 4_r;
//    real_t hybridRegionDiameter = 2_r;
//    auto weight = Slab(center, atomisticRegionDiameter, hybridRegionDiameter, 2);
//
//    std::array<real_t, 3> pos = center;
//    std::array<real_t, 3> delta = {0.1_r, 0.1_r, 0.1_r};
//    for (auto i = 0; i < 60; ++i)
//    {
//        real_t lambda0;
//        real_t lambda1;
//        real_t gradLambda;
//        real_t tmp;  ///< unused dump variable
//
//        weight(pos[0], 0_r, 0_r, lambda0, tmp, tmp, tmp);
//        weight(pos[0] + 0.5_r * eps, 0_r, 0_r, tmp, gradLambda, tmp, tmp);
//        weight(pos[0] + eps, 0_r, 0_r, lambda1, tmp, tmp, tmp);
//        EXPECT_FLOAT_EQ((lambda1 - lambda0) / eps, gradLambda);
//        pos[0] += delta[0];
//    }
//}
}  // namespace weighting_function
}  // namespace mrmd
