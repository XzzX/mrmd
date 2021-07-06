#include "communication/AccumulateForce.hpp"

#include <gtest/gtest.h>

#include <data/Particles.hpp>

namespace mrmd
{
namespace communication
{
namespace impl
{
TEST(AccumulateForceTest, ghostToReal)
{
    // accumulate all force on particle 0
    data::Particles particles(101);
    particles.numLocalParticles = 1;
    particles.numGhostParticles = 100;
    particles.resize(101);

    IndexView correspondingRealParticle("correspondingRealParticle", 101);
    Kokkos::deep_copy(correspondingRealParticle, 0);
    correspondingRealParticle(0) = -1;

    auto force = particles.getForce();
    Cabana::deep_copy(force, 1_r);

    AccumulateForce accumulateForce;
    accumulateForce.ghostToReal(particles, correspondingRealParticle);

    EXPECT_FLOAT_EQ(force(0, 0), 101_r);
    EXPECT_FLOAT_EQ(force(0, 1), 101_r);
    EXPECT_FLOAT_EQ(force(0, 2), 101_r);
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
