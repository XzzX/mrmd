#include "LangevinThermostat.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using LangevinThermostatTest = test::SingleAtom;

TEST_F(LangevinThermostatTest, Simple)
{
    LangevinThermostat langevinThermostat(real_t(0.5), real_t(0.5), real_t(0.1));
    langevinThermostat.apply(atoms);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);
    EXPECT_GE(force(0, 0),
              real_t(9) + langevinThermostat.getPref1() * real_t(1.5) * real_t(7) -
                  langevinThermostat.getPref2() * std::sqrt(real_t(1.5)) * real_t(0.5));
    EXPECT_LE(force(0, 0),
              real_t(9) + langevinThermostat.getPref1() * real_t(1.5) * real_t(7) +
                  langevinThermostat.getPref2() * std::sqrt(real_t(1.5)) * real_t(0.5));
    EXPECT_GE(force(0, 1),
              real_t(7) + langevinThermostat.getPref1() * real_t(1.5) * real_t(5) -
                  langevinThermostat.getPref2() * std::sqrt(real_t(1.5)) * real_t(0.5));
    EXPECT_LE(force(0, 1),
              real_t(7) + langevinThermostat.getPref1() * real_t(1.5) * real_t(5) +
                  langevinThermostat.getPref2() * std::sqrt(real_t(1.5)) * real_t(0.5));
    EXPECT_GE(force(0, 2),
              real_t(8) + langevinThermostat.getPref1() * real_t(1.5) * real_t(3) -
                  langevinThermostat.getPref2() * std::sqrt(real_t(1.5)) * real_t(0.5));
    EXPECT_LE(force(0, 2),
              real_t(8) + langevinThermostat.getPref1() * real_t(1.5) * real_t(3) +
                  langevinThermostat.getPref2() * std::sqrt(real_t(1.5)) * real_t(0.5));
}
}  // namespace action
}  // namespace mrmd