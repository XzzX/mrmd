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
    LangevinThermostat langevinThermostat(0.5_r, 0.5_r, 0.1_r);
    langevinThermostat.apply(atoms);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);
    EXPECT_GE(force(0, 0),
              9_r + langevinThermostat.getPref1() * 1.5_r * 7_r -
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_LE(force(0, 0),
              9_r + langevinThermostat.getPref1() * 1.5_r * 7_r +
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_GE(force(0, 1),
              7_r + langevinThermostat.getPref1() * 1.5_r * 5_r -
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_LE(force(0, 1),
              7_r + langevinThermostat.getPref1() * 1.5_r * 5_r +
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_GE(force(0, 2),
              8_r + langevinThermostat.getPref1() * 1.5_r * 3_r -
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
    EXPECT_LE(force(0, 2),
              8_r + langevinThermostat.getPref1() * 1.5_r * 3_r +
                  langevinThermostat.getPref2() * std::sqrt(1.5_r) * 0.5_r);
}
}  // namespace action
}  // namespace mrmd