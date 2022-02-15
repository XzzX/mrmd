#include "BerendsenThermostat.hpp"

#include <gtest/gtest.h>

#include "analysis/KineticEnergy.hpp"
#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using VelocityScalingTest = test::SingleAtom;

TEST_F(VelocityScalingTest, gamma_0)
{
    constexpr real_t gamma = 0_r;
    constexpr real_t targetTemperature = 3.8_r;
    auto T = analysis::getMeanKineticEnergy(atoms) * (2_r / 3_r);
    action::BerendsenThermostat::apply(atoms, T, targetTemperature, gamma);
    T = analysis::getMeanKineticEnergy(atoms) * (2_r / 3_r);
    EXPECT_FLOAT_EQ(T, 41.5_r);
}

TEST_F(VelocityScalingTest, gamma_1)
{
    constexpr real_t gamma = 1_r;
    constexpr real_t targetTemperature = 3.8_r;
    auto currentTemperature = analysis::getMeanKineticEnergy(atoms) * (2_r / 3_r);
    action::BerendsenThermostat::apply(atoms, currentTemperature, targetTemperature, gamma);
    currentTemperature = analysis::getMeanKineticEnergy(atoms) * (2_r / 3_r);
    EXPECT_FLOAT_EQ(currentTemperature, targetTemperature);
}
}  // namespace action
}  // namespace mrmd