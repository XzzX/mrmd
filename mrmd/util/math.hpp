// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    auto yy = 1_r;

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
    assert(x > 0_r);

    constexpr auto p = 0.3275911_r;
    constexpr auto a1 = 0.254829592_r;
    constexpr auto a2 = -0.284496736_r;
    constexpr auto a3 = 1.421413741_r;
    constexpr auto a4 = -1.453152027_r;
    constexpr auto a5 = 1.061405429_r;
    auto t = 1_r / (1_r + p * x);
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
