#include "Histogram.hpp"

#include <cassert>

namespace mrmd
{
namespace data
{
Histogram& Histogram::operator+=(const Histogram& rhs)
{
    if (numBins != rhs.numBins) exit(-1);
    assert(min == rhs.min);
    assert(max == rhs.max);

    auto policy = Kokkos::RangePolicy<>(0, numBins);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx) { data(idx) += rhs(idx); };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();
    return *this;
}

data::Histogram gradient(const data::Histogram& input, const real_t spacing)
{
    const auto inverseSpacing = 1_r / spacing;
    const auto inverseDoubleSpacing = 1_r / (2_r * spacing);

    data::Histogram grad("gradient", input.min, input.max, input.numBins);
    auto policy = Kokkos::RangePolicy<>(0, input.numBins);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        if (idx == 0)
        {
            grad.data(idx) = (input.data(idx + 1) - input.data(idx)) * inverseSpacing;
            return;
        }

        if (idx == input.numBins - 1)
        {
            grad.data(idx) = (input.data(idx) + input.data(idx - 1)) * inverseSpacing;
            return;
        }

        grad.data(idx) = (input.data(idx + 1) - input.data(idx - 1)) * inverseDoubleSpacing;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    return grad;
}
}  // namespace data
}  // namespace mrmd