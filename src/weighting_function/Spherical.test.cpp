#include "Spherical.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace weighting_function
{
TEST(Spherical, monotonous)
{
    std::array<real_t, 3> center = {2_r, 3_r, 4_r};
    real_t atomisticRadius = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Spherical(center, atomisticRadius, hybridRegionDiameter);

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

TEST(Spherical, boundaryValues)
{
    std::array<real_t, 3> center = {2_r, 3_r, 4_r};
    real_t atomisticRadius = 2_r;
    real_t hybridRegionDiameter = 2_r;
    auto weight = Spherical(center, atomisticRadius, hybridRegionDiameter);

    auto w = weight(2.1_r, 3.1_r, 4.1_r);
    EXPECT_FLOAT_EQ(w, 1_r);

    w = weight(6.1_r, 7.1_r, 8.1_r);
    EXPECT_FLOAT_EQ(w, 0_r);
}
}  // namespace weighting_function
}  // namespace mrmd
