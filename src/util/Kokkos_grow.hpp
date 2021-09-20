#pragma once

#include <Kokkos_View.hpp>
#include <cassert>

namespace mrmd
{
namespace util
{
template <class T, class... P>
void grow(Kokkos::View<T, P...>& view, idx_t size, real_t safetyMargin = 1.1_r)
{
    assert(size >= 0);

    if (view.extent(0) <= static_cast<unsigned long>(size))
    {
        Kokkos::resize(view, idx_c(real_c(size) * safetyMargin));
    }

    assert(view.extent(0) >= size);
}
}  // namespace util
}  // namespace mrmd
