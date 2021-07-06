#include "VelocityVerlet.hpp"

#include <gtest/gtest.h>

namespace action
{
class VelocityVerletTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto pos = particles.getPos();
        auto vel = particles.getVel();
        auto force = particles.getForce();

        pos(0, 0) = 1_r;
        pos(0, 1) = 2_r;
        pos(0, 2) = 3_r;

        vel(0, 0) = 4_r;
        vel(0, 1) = 5_r;
        vel(0, 2) = 6_r;

        force(0, 0) = 7_r;
        force(0, 1) = 8_r;
        force(0, 2) = 9_r;

        particles.numLocalParticles = 1;
    }

    // void TearDown() override {}

    data::Particles particles = data::Particles(1);
};

TEST_F(VelocityVerletTest, preForceIntegration)
{
    VelocityVerlet vv(4_r);
    vv.preForceIntegrate(particles);

    EXPECT_FLOAT_EQ(particles.getForce()(0, 0), 7_r);
    EXPECT_FLOAT_EQ(particles.getForce()(0, 1), 8_r);
    EXPECT_FLOAT_EQ(particles.getForce()(0, 2), 9_r);

    EXPECT_FLOAT_EQ(particles.getVel()(0, 0), 18_r);
    EXPECT_FLOAT_EQ(particles.getVel()(0, 1), 21_r);
    EXPECT_FLOAT_EQ(particles.getVel()(0, 2), 24_r);

    EXPECT_FLOAT_EQ(particles.getPos()(0, 0), 73_r);
    EXPECT_FLOAT_EQ(particles.getPos()(0, 1), 86_r);
    EXPECT_FLOAT_EQ(particles.getPos()(0, 2), 99_r);
}

TEST_F(VelocityVerletTest, postForceIntegration)
{
    VelocityVerlet vv(4_r);
    vv.postForceIntegrate(particles);

    EXPECT_FLOAT_EQ(particles.getForce()(0, 0), 7_r);
    EXPECT_FLOAT_EQ(particles.getForce()(0, 1), 8_r);
    EXPECT_FLOAT_EQ(particles.getForce()(0, 2), 9_r);

    EXPECT_FLOAT_EQ(particles.getVel()(0, 0), 18_r);
    EXPECT_FLOAT_EQ(particles.getVel()(0, 1), 21_r);
    EXPECT_FLOAT_EQ(particles.getVel()(0, 2), 24_r);

    EXPECT_FLOAT_EQ(particles.getPos()(0, 0), 1_r);
    EXPECT_FLOAT_EQ(particles.getPos()(0, 1), 2_r);
    EXPECT_FLOAT_EQ(particles.getPos()(0, 2), 3_r);
}

}  // namespace action