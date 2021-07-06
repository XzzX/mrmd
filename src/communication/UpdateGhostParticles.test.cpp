#include "communication/UpdateGhostParticles.hpp"

#include <gtest/gtest.h>

#include <data/Particles.hpp>

namespace mrmd
{
namespace communication
{
namespace impl
{
struct UpdateGhostParticlesTestData
{
    std::array<real_t, 3> initialDelta;
    std::array<real_t, 3> finalDelta;
};

std::ostream& operator<<(std::ostream& os, const UpdateGhostParticlesTestData& data)
{
    os << "initial delta: [" << data.initialDelta[0] << ", " << data.initialDelta[1] << ", "
       << data.initialDelta[2] << "], ";
    os << "mapped delta: [" << data.finalDelta[0] << ", " << data.finalDelta[1] << ", "
       << data.finalDelta[2] << "]" << std::endl;
    return os;
}

class UpdateGhostParticlesTest : public testing::TestWithParam<UpdateGhostParticlesTestData>
{
protected:
    void SetUp() override
    {
        auto pos = particles.getPos();
        pos(0, 0) = 0.5_r;
        pos(0, 1) = 0.5_r;
        pos(0, 2) = 0.5_r;
        particles.numLocalParticles = 1;
        particles.numGhostParticles = 1;

        correspondingRealParticle = IndexView("correspondingRealParticle", 2);
        correspondingRealParticle(0) = -1;
        correspondingRealParticle(1) = 0;
    }

    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, 0.1_r);
    data::Particles particles = data::Particles(2);
    IndexView correspondingRealParticle;
};

TEST_P(UpdateGhostParticlesTest, Check)
{
    auto pos = particles.getPos();
    pos(1, 0) = 0.5_r + GetParam().initialDelta[0];
    pos(1, 1) = 0.5_r + GetParam().initialDelta[1];
    pos(1, 2) = 0.5_r + GetParam().initialDelta[2];

    UpdateGhostParticles updateGhostParticles(subdomain);
    updateGhostParticles.updateOnlyPos(particles, correspondingRealParticle);

    EXPECT_FLOAT_EQ(pos(1, 0), 0.5_r + GetParam().finalDelta[0]);
    EXPECT_FLOAT_EQ(pos(1, 1), 0.5_r + GetParam().finalDelta[1]);
    EXPECT_FLOAT_EQ(pos(1, 2), 0.5_r + GetParam().finalDelta[2]);
}

INSTANTIATE_TEST_SUITE_P(
    ShiftX,
    UpdateGhostParticlesTest,
    testing::Values(UpdateGhostParticlesTestData{{0.2_r, 0_r, 0_r}, {1_r, 0_r, 0_r}},
                    UpdateGhostParticlesTestData{{-0.2_r, 0_r, 0_r}, {-1_r, 0_r, 0_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftY,
    UpdateGhostParticlesTest,
    testing::Values(UpdateGhostParticlesTestData{{0_r, 0.2_r, 0_r}, {0_r, 1_r, 0_r}},
                    UpdateGhostParticlesTestData{{0_r, -0.2_r, 0_r}, {0_r, -1_r, 0_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftZ,
    UpdateGhostParticlesTest,
    testing::Values(UpdateGhostParticlesTestData{{0_r, 0_r, 0.2_r}, {0_r, 0_r, 1_r}},
                    UpdateGhostParticlesTestData{{0_r, 0_r, -0.2_r}, {0_r, 0_r, -1_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftXYZ,
    UpdateGhostParticlesTest,
    testing::Values(UpdateGhostParticlesTestData{{0.2_r, 0.2_r, 0.2_r}, {1_r, 1_r, 1_r}},
                    UpdateGhostParticlesTestData{{-0.2_r, -0.2_r, -0.2_r}, {-1_r, -1_r, -1_r}}));

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
