#include "BerendsenBarostat.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace action
{
using BerendsenBarostatTest = test::SingleAtom;

TEST_F(BerendsenBarostatTest, gamma_0)
{
    constexpr real_t gamma = real_t(0);
    constexpr real_t currentPressure = real_t(1);
    constexpr real_t targetPressure = real_t(3.8);
    data::Subdomain subdomain(
        {real_t(0), real_t(0), real_t(0)}, {real_t(1), real_t(1), real_t(1)}, real_t(0.1));
    action::BerendsenBarostat::apply(atoms, currentPressure, targetPressure, gamma, subdomain);

    EXPECT_FLOAT_EQ(subdomain.maxCorner[0], real_t(1));
    EXPECT_FLOAT_EQ(subdomain.maxCorner[1], real_t(1));
    EXPECT_FLOAT_EQ(subdomain.maxCorner[2], real_t(1));

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);

    EXPECT_FLOAT_EQ(pos(0, 0), real_t(2));
    EXPECT_FLOAT_EQ(pos(0, 1), real_t(3));
    EXPECT_FLOAT_EQ(pos(0, 2), real_t(4));
}

TEST_F(BerendsenBarostatTest, gamma_1)
{
    constexpr real_t gamma = real_t(1);
    constexpr real_t currentPressure = real_t(2);
    constexpr real_t targetPressure = real_t(1);
    data::Subdomain subdomain(
        {real_t(0), real_t(0), real_t(0)}, {real_t(1), real_t(1), real_t(1)}, real_t(0.1));
    action::BerendsenBarostat::apply(atoms, currentPressure, targetPressure, gamma, subdomain);

    EXPECT_FLOAT_EQ(subdomain.maxCorner[0], std::cbrt(real_t(2)));
    EXPECT_FLOAT_EQ(subdomain.maxCorner[1], std::cbrt(real_t(2)));
    EXPECT_FLOAT_EQ(subdomain.maxCorner[2], std::cbrt(real_t(2)));

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);

    EXPECT_FLOAT_EQ(pos(0, 0), real_t(2) * std::cbrt(real_t(2)));
    EXPECT_FLOAT_EQ(pos(0, 1), real_t(3) * std::cbrt(real_t(2)));
    EXPECT_FLOAT_EQ(pos(0, 2), real_t(4) * std::cbrt(real_t(2)));
}
}  // namespace action
}  // namespace mrmd