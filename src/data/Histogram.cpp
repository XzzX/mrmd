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

    auto lhs = data;
    auto policy = Kokkos::RangePolicy<>(0, numBins);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx) { lhs(idx) += rhs.data(idx); };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();
    return *this;
}

data::Histogram gradient(const data::Histogram& input)
{
    const auto inverseSpacing = input.inverseBinSize;
    const auto inverseDoubleSpacing = 0.5_r * input.inverseBinSize;

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
            grad.data(idx) = (input.data(idx) - input.data(idx - 1)) * inverseSpacing;
            return;
        }

        grad.data(idx) = (input.data(idx + 1) - input.data(idx - 1)) * inverseDoubleSpacing;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    return grad;
}

std::ostream& operator<<(std::ostream& os, const data::Histogram& hist)
{
    auto hData = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hist.data);
    for (auto i = 0; i < hist.numBins; ++i)
    {
        os << hData(i) << " ";
    }
    return os;
}

}  // namespace data
}  // namespace mrmd