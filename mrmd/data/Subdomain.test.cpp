#include "Subdomain.hpp"

#include <gtest/gtest.h>

#include "datatypes.hpp"

namespace mrmd
{
TEST(Subdomain, Constructor)
{
    data::Subdomain subdomain({1, 2, 3}, {3, 6, 9}, real_t(0.5));

    EXPECT_FLOAT_EQ(subdomain.minCorner[0], real_t(1));
    EXPECT_FLOAT_EQ(subdomain.minCorner[1], real_t(2));
    EXPECT_FLOAT_EQ(subdomain.minCorner[2], real_t(3));

    EXPECT_FLOAT_EQ(subdomain.maxCorner[0], real_t(3));
    EXPECT_FLOAT_EQ(subdomain.maxCorner[1], real_t(6));
    EXPECT_FLOAT_EQ(subdomain.maxCorner[2], real_t(9));

    EXPECT_FLOAT_EQ(subdomain.ghostLayerThickness, real_t(0.5));

    EXPECT_FLOAT_EQ(subdomain.diameter[0], real_t(2));
    EXPECT_FLOAT_EQ(subdomain.diameter[1], real_t(4));
    EXPECT_FLOAT_EQ(subdomain.diameter[2], real_t(6));

    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[0], real_t(3));
    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[1], real_t(5));
    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[2], real_t(7));

    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[0], real_t(1.5));
    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[1], real_t(2.5));
    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[2], real_t(3.5));

    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[0], real_t(2.5));
    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[1], real_t(5.5));
    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[2], real_t(8.5));

    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[0], real_t(0.5));
    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[1], real_t(1.5));
    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[2], real_t(2.5));

    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[0], real_t(3.5));
    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[1], real_t(6.5));
    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[2], real_t(9.5));
}

TEST(Subdomain, scale)
{
    auto scalingFactor = real_t(0.5);
    data::Subdomain subdomain({1, 2, 3}, {3, 6, 9}, real_t(0.2));
    subdomain.scale(scalingFactor);

    EXPECT_FLOAT_EQ(subdomain.minCorner[0], real_t(1) * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.minCorner[1], real_t(2) * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.minCorner[2], real_t(3) * scalingFactor);

    EXPECT_FLOAT_EQ(subdomain.maxCorner[0], real_t(3) * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.maxCorner[1], real_t(6) * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.maxCorner[2], real_t(9) * scalingFactor);

    EXPECT_FLOAT_EQ(subdomain.ghostLayerThickness, real_t(0.2));

    EXPECT_FLOAT_EQ(subdomain.diameter[0], real_t(2) * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.diameter[1], real_t(4) * scalingFactor);
    EXPECT_FLOAT_EQ(subdomain.diameter[2], real_t(6) * scalingFactor);

    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[0], real_t(2) * scalingFactor + real_t(0.4));
    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[1], real_t(4) * scalingFactor + real_t(0.4));
    EXPECT_FLOAT_EQ(subdomain.diameterWithGhostLayer[2], real_t(6) * scalingFactor + real_t(0.4));

    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[0], real_t(0.7));
    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[1], real_t(1.2));
    EXPECT_FLOAT_EQ(subdomain.minInnerCorner[2], real_t(1.7));

    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[0], real_t(1.3));
    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[1], real_t(2.8));
    EXPECT_FLOAT_EQ(subdomain.maxInnerCorner[2], real_t(4.3));

    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[0], real_t(0.3));
    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[1], real_t(0.8));
    EXPECT_FLOAT_EQ(subdomain.minGhostCorner[2], real_t(1.3));

    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[0], real_t(1.7));
    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[1], real_t(3.2));
    EXPECT_FLOAT_EQ(subdomain.maxGhostCorner[2], real_t(4.7));
}
}  // namespace mrmd