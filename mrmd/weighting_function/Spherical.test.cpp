#include "Spherical.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace weighting_function
{
TEST(Spherical, monotonous)
{
    std::array<real_t, 3> center = {real_t(2), real_t(3), real_t(4)};
    real_t atomisticRadius = real_t(2);
    real_t hybridRegionDiameter = real_t(2);
    auto weight = Spherical(center, atomisticRadius, hybridRegionDiameter, 2);

    std::array<real_t, 3> pos = center;
    std::array<real_t, 3> delta = {real_t(0.1), real_t(0.1), real_t(0.1)};
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
    std::array<real_t, 3> center = {real_t(2), real_t(3), real_t(4)};
    real_t atomisticRadius = real_t(2);
    real_t hybridRegionDiameter = real_t(2);
    auto weight = Spherical(center, atomisticRadius, hybridRegionDiameter, 2);

    auto w = weight(real_t(2.1), real_t(3.1), real_t(4.1));
    EXPECT_FLOAT_EQ(w, real_t(1));

    w = weight(real_t(6.1), real_t(7.1), real_t(8.1));
    EXPECT_FLOAT_EQ(w, real_t(0));
}

TEST(Spherical, derivative)
{
    const auto eps = real_t(1e-10);
    std::array<real_t, 3> center = {real_t(2), real_t(3), real_t(4)};
    real_t atomisticRadius = real_t(2);
    real_t hybridRegionDiameter = real_t(2);
    auto weight = Spherical(center, atomisticRadius, hybridRegionDiameter, 2);

    std::array<real_t, 3> pos = center;
    std::array<real_t, 3> delta = {real_t(0.1), real_t(0.1), real_t(0.1)};
    for (auto i = 0; i < 60; ++i)
    {
        real_t lambda0;
        real_t gradLambdaX0;
        real_t gradLambdaY0;
        real_t gradLambdaZ0;

        real_t lambda1;
        real_t gradLambdaX1;
        real_t gradLambdaY1;
        real_t gradLambdaZ1;

        weight(pos[0], real_t(0), real_t(0), lambda0, gradLambdaX0, gradLambdaY0, gradLambdaZ0);
        weight(
            pos[0] + eps, real_t(0), real_t(0), lambda1, gradLambdaX1, gradLambdaY1, gradLambdaZ1);
        EXPECT_FLOAT_EQ((lambda1 - lambda0) / eps, real_t(0.5) * (gradLambdaX0 + gradLambdaX1));
        pos[0] += delta[0];
    }
}
}  // namespace weighting_function
}  // namespace mrmd
