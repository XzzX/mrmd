#include "MultiHistogram.hpp"

#include <cassert>

namespace mrmd
{
namespace data
{
MultiHistogram& MultiHistogram::operator+=(const MultiHistogram& rhs)
{
    if (numBins != rhs.numBins) exit(EXIT_FAILURE);
    assert(min == rhs.min);
    assert(max == rhs.max);

    auto lhs = data;
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numBins, numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx)
    {
        lhs(idx, jdx) += rhs.data(idx, jdx);
    };
    Kokkos::parallel_for("MultiHistogram::operator+=", policy, kernel);
    Kokkos::fence();
    return *this;
}

void MultiHistogram::scale(const real_t& scalingFactor)
{
    auto hist = data;  // avoid capturing this pointer
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({idx_t(0), idx_t(0)}, {numBins, numHistograms});
    auto normalizeSampleKernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx)
    {
        hist(idx, jdx) *= scalingFactor;
    };
    Kokkos::parallel_for(policy, normalizeSampleKernel, "MultiHistogram::scale");
    Kokkos::fence();
}

void MultiHistogram::scale(const ScalarView& scalingFactor)
{
    CHECK_GREATEREQUAL(idx_c(scalingFactor.extent(0)), numHistograms);

    auto hist = data;  // avoid capturing this pointer
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({idx_t(0), idx_t(0)}, {numBins, numHistograms});
    auto normalizeSampleKernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx)
    {
        hist(idx, jdx) *= scalingFactor(jdx);
    };
    Kokkos::parallel_for(policy, normalizeSampleKernel, "MultiHistogram::scale");
    Kokkos::fence();
}

data::MultiHistogram gradient(const data::MultiHistogram& input)
{
    const auto inverseSpacing = input.inverseBinSize;
    const auto inverseDoubleSpacing = 0.5_r * input.inverseBinSize;

    data::MultiHistogram grad("gradient", input.min, input.max, input.numBins, input.numHistograms);
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {input.numBins, input.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx)
    {
        if (idx == 0)
        {
            grad.data(idx, jdx) =
                (input.data(idx + 1, jdx) - input.data(idx, jdx)) * inverseSpacing;
            return;
        }

        if (idx == input.numBins - 1)
        {
            grad.data(idx, jdx) =
                (input.data(idx, jdx) - input.data(idx - 1, jdx)) * inverseSpacing;
            return;
        }

        grad.data(idx, jdx) =
            (input.data(idx + 1, jdx) - input.data(idx - 1, jdx)) * inverseDoubleSpacing;
    };
    Kokkos::parallel_for("MultiHistogram::gradient", policy, kernel);
    Kokkos::fence();

    return grad;
}

}  // namespace data
}  // namespace mrmd