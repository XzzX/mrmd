#pragma once

#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
KOKKOS_INLINE_FUNCTION
real_t dot3(real_t const* const a, real_t const* const b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

KOKKOS_INLINE_FUNCTION real_t sqr(const real_t& sqr) { return sqr * sqr; }

KOKKOS_INLINE_FUNCTION real_t powInt(const real_t& x, const idx_t n)
{
    auto ww = x;
    auto yy = real_t(1);

    for (idx_t nn = (n > 0) ? n : -n; nn != 0; nn >>= 1)
    {
        if ((nn & 1) == 1)
        {
            yy *= ww;
        }
        ww *= ww;
    }

    return (n > 0) ? yy : 1.0 / yy;
}

/**
 * "Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Table"
 * Milton Abramowitz (1970)
 * "Rational Approximations", equation 7.1.27:
 *
 * accuracy: 1.5x10^-7
 *
 * @param expX2 intermediate value std::exp(-x * x)
 */
KOKKOS_INLINE_FUNCTION real_t approxErfc(const real_t& x, real_t& expX2)
{
    assert(x > real_t(0));

    constexpr auto p = real_t(0.3275911);
    constexpr auto a1 = real_t(0.254829592);
    constexpr auto a2 = real_t(-0.284496736);
    constexpr auto a3 = real_t(1.421413741);
    constexpr auto a4 = real_t(-1.453152027);
    constexpr auto a5 = real_t(1.061405429);
    auto t = real_t(1) / (real_t(1) + p * x);
    expX2 = std::exp(-x * x);
    return t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * expX2;
}

KOKKOS_INLINE_FUNCTION real_t approxErfc(const real_t& x)
{
    [[maybe_unused]] real_t tmp;
    return approxErfc(x, tmp);
}

}  // namespace util
}  // namespace mrmd
