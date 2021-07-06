#include "LangevinThermostat.hpp"

#include <gtest/gtest.h>

#include "data/Particles.hpp"

namespace action
{
TEST(LangevinThermostat, Simple)
{
    data::Particles particles(1);
    auto vel = particles.getVel();
    vel(0, 0) = +2_r;
    vel(0, 1) = +3_r;
    vel(0, 2) = +4_r;
    auto force = particles.getForce();
    force(0, 0) = +2_r;
    force(0, 1) = +3_r;
    force(0, 2) = +4_r;
    particles.numLocalParticles = 1;
    particles.numGhostParticles = 0;

    LangevinThermostat langevinThermostat(0.5_r, 0.5_r, 0.1_r);
    langevinThermostat.applyThermostat(particles);

    EXPECT_GE(force(0, 0), 1_r - langevinThermostat.pref2 * 0.5_r);
    EXPECT_LE(force(0, 0), 1_r + langevinThermostat.pref2 * 0.5_r);

    EXPECT_GE(force(0, 1), 1.5_r - langevinThermostat.pref2 * 0.5_r);
    EXPECT_LE(force(0, 1), 1.5_r + langevinThermostat.pref2 * 0.5_r);

    EXPECT_GE(force(0, 2), 2_r - langevinThermostat.pref2 * 0.5_r);
    EXPECT_LE(force(0, 2), 2_r + langevinThermostat.pref2 * 0.5_r);
}
}  // namespace action