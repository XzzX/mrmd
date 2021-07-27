#pragma once

#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
KOKKOS_INLINE_FUNCTION real_t sqr(const real_t& sqr) { return sqr * sqr; }
KOKKOS_INLINE_FUNCTION real_t powInt(const real_t& x, const int n)
{
    auto ww = x;
    auto yy = 1_r;

    for (int nn = (n > 0) ? n : -n; nn != 0; nn >>= 1)
    {
        if ((nn & 1) == 1) yy *= ww;
        ww *= ww;
    }

    return (n > 0) ? yy : 1.0 / yy;
}

}  // namespace util
}  // namespace mrmd
