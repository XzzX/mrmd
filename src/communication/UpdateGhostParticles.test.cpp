#include "communication/UpdateGhostParticles.hpp"

#include <gtest/gtest.h>

#include <data/Particles.hpp>
namespace communication
{
namespace impl
{
TEST(UpdateGhostParticlesTest, updateOnlyPos)
{
    // accumulate all force on particle 0
    Particles particles(2);
    particles.numLocalParticles = 1;
    particles.numGhostParticles = 1;
    particles.resize(2);

    IndexView correspondingRealParticle("correspondingRealParticle", 2);
    Kokkos::deep_copy(correspondingRealParticle, 0);
    correspondingRealParticle(0) = -1;

    auto pos = particles.getPos();
    Cabana::deep_copy(pos, 2_r);
    pos(1, 0) = 3_r;
    pos(1, 1) = 3_r;
    pos(1, 2) = 3_r;

    UpdateGhostParticles updateGhostParticles;
    updateGhostParticles.updateOnlyPos(particles, correspondingRealParticle);

    EXPECT_FLOAT_EQ(pos(1, 0), 2_r);
    EXPECT_FLOAT_EQ(pos(1, 1), 2_r);
    EXPECT_FLOAT_EQ(pos(1, 2), 2_r);
}

}  // namespace impl
}  // namespace communication
