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
    constexpr real_t gamma = real_t(0);
    constexpr real_t targetTemperature = real_t(3.8);
    auto T = analysis::getMeanKineticEnergy(atoms) * (real_t(2) / real_t(3));
    action::BerendsenThermostat::apply(atoms, T, targetTemperature, gamma);
    T = analysis::getMeanKineticEnergy(atoms) * (real_t(2) / real_t(3));
    EXPECT_FLOAT_EQ(T, real_t(41.5));
}

TEST_F(VelocityScalingTest, gamma_1)
{
    constexpr real_t gamma = real_t(1);
    constexpr real_t targetTemperature = real_t(3.8);
    auto currentTemperature = analysis::getMeanKineticEnergy(atoms) * (real_t(2) / real_t(3));
    action::BerendsenThermostat::apply(atoms, currentTemperature, targetTemperature, gamma);
    currentTemperature = analysis::getMeanKineticEnergy(atoms) * (real_t(2) / real_t(3));
    EXPECT_FLOAT_EQ(currentTemperature, targetTemperature);
}
}  // namespace action
}  // namespace mrmd