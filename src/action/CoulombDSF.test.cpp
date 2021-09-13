#include "CoulombDSF.hpp"

#include <gtest/gtest.h>

#include <fstream>

namespace mrmd
{
namespace action
{
constexpr real_t coulombEnergy(const real_t r, const real_t q1, const real_t q2)
{
    constexpr real_t C = 1_r;
    constexpr real_t epsilon = 1_r;
    return C * q1 * q1 / (epsilon * r);
}

constexpr real_t coulombEnergyDSF(
    const real_t r, const real_t q1, const real_t q2, const real_t alpha, const real_t rc)
{
    return q1 * q1 *
           (std::erfc(alpha * r) / r - std::erfc(alpha * rc) / rc +
            (std::erfc(alpha * rc) / (rc * rc) +
             2_r * alpha / M_SQRTPI * std::exp(-alpha * alpha * rc * rc) / rc) *
                (r - rc));
}

real_t coulombForceDSF(
    const real_t r, const real_t q1, const real_t q2, const real_t alpha, const real_t rc)
{
    // DOI: 10.1140/epjst/e2016-60151-6
    // equation 16

    real_t prefac = 138.935458_r * q1 * q2;
    auto distSqr = r * r;
    auto rcSqr = rc * rc;
    auto bracket = 0_r;
    bracket += std::erfc(alpha * r) / distSqr;
    bracket += 2_r * alpha / M_SQRTPI * std::exp(-alpha * alpha * distSqr) / r;
    bracket -= std::erfc(alpha * rc) / rcSqr;
    bracket -= 2_r * alpha / M_SQRTPI * std::exp(-alpha * alpha * rcSqr) / rc;
    return prefac * bracket / r;
}

TEST(CoulombDSF, ForceExplicitComparison)
{
    constexpr real_t rc = 5_r;
    constexpr real_t alpha = 0.1_r;
    constexpr real_t q1 = +1.2_r;
    constexpr real_t q2 = -1.3_r;
    impl::CoulombDSF coulomb(rc, alpha);

    for (real_t x = 1e-8_r; x < rc; x += 0.01_r)
    {
        auto relativeError =
            std::abs((coulomb.computeForce(x * x, q1, q2) - coulombForceDSF(x, q1, q2, alpha, rc)) /
                     coulombForceDSF(x, q1, q2, alpha, rc));
        EXPECT_LT(relativeError, 1e-4_r);
    }
}

TEST(CoulombDSF, Symmetry)
{
    constexpr real_t rc = 1_r;
    constexpr real_t alpha = 0.1_r;
    constexpr real_t q = 1_r;
    constexpr real_t distSqr = 1_r;
    impl::CoulombDSF coulomb(rc, alpha);

    EXPECT_FLOAT_EQ(coulomb.computeForce(distSqr, +q, +q), coulomb.computeForce(distSqr, -q, -q));
    EXPECT_FLOAT_EQ(coulomb.computeForce(distSqr, +q, -q), coulomb.computeForce(distSqr, -q, +q));
    EXPECT_FLOAT_EQ(coulomb.computeForce(distSqr, +q, +q), -coulomb.computeForce(distSqr, +q, -q));
}

TEST(CoulombDSF, shift)
{
    constexpr real_t rc = 1.5_r;
    constexpr real_t alpha = 0.1_r;
    constexpr real_t q = 1_r;
    impl::CoulombDSF coulomb(rc, alpha);

    EXPECT_NEAR(coulomb.computeForce(rc * rc, +q, +q), 0_r, 1e-5_r);
}

}  // namespace action
}  // namespace mrmd