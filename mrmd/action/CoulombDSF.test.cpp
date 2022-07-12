#include "CoulombDSF.hpp"

#include <gtest/gtest.h>

#include <fstream>

namespace mrmd
{
namespace action
{
constexpr real_t coulombEnergy(const real_t r, const real_t q1, const real_t q2)
{
    constexpr real_t C = real_t(1);
    constexpr real_t epsilon = real_t(1);
    return C * q1 * q1 / (epsilon * r);
}

constexpr real_t coulombEnergyDSF(
    const real_t r, const real_t q1, const real_t q2, const real_t alpha, const real_t rc)
{
    real_t prefac = real_t(138.935458) * q1 * q2;
    auto bracket = real_t(0);
    bracket += std::erfc(alpha * r) / r;
    bracket -= std::erfc(alpha * rc) / rc;
    bracket += (std::erfc(alpha * rc) / (rc * rc) +
                real_t(2) * alpha / M_SQRTPI * std::exp(-alpha * alpha * rc * rc) / rc) *
               (r - rc);
    return prefac * bracket;
}

real_t coulombForceDSF(
    const real_t r, const real_t q1, const real_t q2, const real_t alpha, const real_t rc)
{
    // DOI: 10.1140/epjst/e2016-60151-6
    // equation 16

    real_t prefac = real_t(138.935458) * q1 * q2;
    auto distSqr = r * r;
    auto rcSqr = rc * rc;
    auto bracket = real_t(0);
    bracket += std::erfc(alpha * r) / distSqr;
    bracket += real_t(2) * alpha / M_SQRTPI * std::exp(-alpha * alpha * distSqr) / r;
    bracket -= std::erfc(alpha * rc) / rcSqr;
    bracket -= real_t(2) * alpha / M_SQRTPI * std::exp(-alpha * alpha * rcSqr) / rc;
    return prefac * bracket / r;
}

TEST(CoulombDSF, EnergyExplicitComparison)
{
    constexpr real_t rc = real_t(5);
    constexpr real_t alpha = real_t(0.1);
    constexpr real_t q1 = real_t(+1.2);
    constexpr real_t q2 = real_t(-1.3);
    impl::CoulombDSF coulomb(rc, alpha);

    for (real_t x = real_t(1e-8); x < rc - real_t(1); x += real_t(0.01))
    {
        auto relativeError = std::abs(
            (coulomb.computeEnergy(x * x, q1, q2) - coulombEnergyDSF(x, q1, q2, alpha, rc)) /
            coulombEnergyDSF(x, q1, q2, alpha, rc));
        EXPECT_LT(relativeError, real_t(1e-5));
    }
}

TEST(CoulombDSF, ForceExplicitComparison)
{
    constexpr real_t rc = real_t(5);
    constexpr real_t alpha = real_t(0.1);
    constexpr real_t q1 = real_t(+1.2);
    constexpr real_t q2 = real_t(-1.3);
    impl::CoulombDSF coulomb(rc, alpha);

    for (real_t x = real_t(1e-8); x < rc; x += real_t(0.01))
    {
        auto relativeError =
            std::abs((coulomb.computeForce(x * x, q1, q2) - coulombForceDSF(x, q1, q2, alpha, rc)) /
                     coulombForceDSF(x, q1, q2, alpha, rc));
        EXPECT_LT(relativeError, real_t(1e-4));
    }
}

TEST(CoulombDSF, Symmetry)
{
    constexpr real_t rc = real_t(1);
    constexpr real_t alpha = real_t(0.1);
    constexpr real_t q = real_t(1);
    constexpr real_t distSqr = real_t(1);
    impl::CoulombDSF coulomb(rc, alpha);

    EXPECT_FLOAT_EQ(coulomb.computeForce(distSqr, +q, +q), coulomb.computeForce(distSqr, -q, -q));
    EXPECT_FLOAT_EQ(coulomb.computeForce(distSqr, +q, -q), coulomb.computeForce(distSqr, -q, +q));
    EXPECT_FLOAT_EQ(coulomb.computeForce(distSqr, +q, +q), -coulomb.computeForce(distSqr, +q, -q));
}

TEST(CoulombDSF, shift)
{
    constexpr real_t rc = real_t(1.5);
    constexpr real_t alpha = real_t(0.1);
    constexpr real_t q = real_t(1);
    impl::CoulombDSF coulomb(rc, alpha);

    EXPECT_NEAR(coulomb.computeForce(rc * rc, +q, +q), real_t(0), real_t(1e-5));
}

}  // namespace action
}  // namespace mrmd