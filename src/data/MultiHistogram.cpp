#include "MultiHistogram.hpp"

#include <cassert>

#include "util/math.hpp"

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

MultiHistogram& MultiHistogram::operator/=(const MultiHistogram& rhs)
{
    if (numBins != rhs.numBins) exit(EXIT_FAILURE);
    assert(min == rhs.min);
    assert(max == rhs.max);

    auto lhs = data;
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numBins, numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx)
    {
        lhs(idx, jdx) /= rhs.data(idx, jdx);
    };
    Kokkos::parallel_for("MultiHistogram::operator/=", policy, kernel);
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
    MRMD_HOST_CHECK_GREATEREQUAL(idx_c(scalingFactor.extent(0)), numHistograms);

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

void MultiHistogram::makeSymmetric()
{
    auto maxIdx = numBins - 1;
    auto hist = data;  // avoid capturing this pointer
    auto policy =
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({idx_t(0), idx_t(0)}, {numBins / 2, numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx)
    {
        auto val = 0.5_r * (hist(idx, jdx) + hist(maxIdx - idx, jdx));
        hist(idx, jdx) = val;
        hist(maxIdx - idx, jdx) = val;
    };
    Kokkos::parallel_for("MultiHistogram::makeSymmetric", policy, kernel);
    Kokkos::fence();
}

void cumulativeMovingAverage(data::MultiHistogram& average,
                             const data::MultiHistogram& current,
                             const real_t movingAverageFactor)
{
    MRMD_HOST_CHECK_EQUAL(average.numBins, current.numBins);
    MRMD_HOST_CHECK_EQUAL(average.numHistograms, current.numHistograms);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({idx_t(0), idx_t(0)},
                                                         {average.numBins, average.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        // use running average to calculate new mean compensation energy
        average.data(binIdx, histogramIdx) =
            (movingAverageFactor * average.data(binIdx, histogramIdx) +
             current.data(binIdx, histogramIdx)) /
            (movingAverageFactor + 1_r);
    };
    Kokkos::parallel_for("MultiHistogram::cumulativeMovingAverage", policy, kernel);
    Kokkos::fence();
}

data::MultiHistogram gradient(const data::MultiHistogram& input, bool periodic)
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
            if (periodic)
            {
                grad.data(idx, jdx) =
                    (input.data(idx + 1, jdx) - input.data(input.numBins - 1, jdx)) *
                    inverseDoubleSpacing;
            }
            else
            {
                grad.data(idx, jdx) =
                    (input.data(idx + 1, jdx) - input.data(idx, jdx)) * inverseSpacing;
            }
            return;
        }

        if (idx == input.numBins - 1)
        {
            if (periodic)
            {
                grad.data(idx, jdx) =
                    (input.data(0, jdx) - input.data(idx - 1, jdx)) * inverseDoubleSpacing;
            }
            else
            {
                grad.data(idx, jdx) =
                    (input.data(idx, jdx) - input.data(idx - 1, jdx)) * inverseSpacing;
            }
            return;
        }

        grad.data(idx, jdx) =
            (input.data(idx + 1, jdx) - input.data(idx - 1, jdx)) * inverseDoubleSpacing;
    };
    Kokkos::parallel_for("MultiHistogram::gradient", policy, kernel);
    Kokkos::fence();

    return grad;
}

data::MultiHistogram smoothen(data::MultiHistogram& input,
                              const real_t& sigma,
                              const real_t& range,
                              const bool periodic)
{
    const auto inverseSigma = 1_r / sigma;
    /// how many neighboring bins are affected
    const idx_t delta = int_c(range * sigma * input.inverseBinSize);

    data::MultiHistogram smoothenedDensityProfile(
        "smooth-input", input.min, input.max, input.numBins, input.numHistograms);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({idx_t(0), idx_t(0)},
                                                         {input.numBins, input.numHistograms});
    auto kernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        auto normalization = 0_r;

        idx_t jdxMin = binIdx - delta;
        idx_t jdxMax = binIdx + delta;

        if (!periodic)
        {
            jdxMin = std::max(idx_t(0), jdxMin);
            jdxMax = std::min(input.numBins - 1, jdxMax);
        }
        assert(jdxMin <= jdxMax);

        for (auto jdx = jdxMin; jdx <= jdxMax; ++jdx)
        {
            auto possiblyMappedIdx = jdx;
            if (!periodic)
            {
                if (possiblyMappedIdx < 0) possiblyMappedIdx += input.numBins;
                if (possiblyMappedIdx >= input.numBins) possiblyMappedIdx -= input.numBins;
            }
            const auto eFunc =
                std::exp(-util::sqr(real_c(binIdx - jdx) * input.binSize * inverseSigma));
            normalization += eFunc;
            smoothenedDensityProfile.data(binIdx, histogramIdx) +=
                input.data(possiblyMappedIdx, histogramIdx) * eFunc;
        }

        smoothenedDensityProfile.data(binIdx, histogramIdx) /= normalization;
    };
    Kokkos::parallel_for("MultiHistogram::smoothen", policy, kernel);
    Kokkos::fence();

    return smoothenedDensityProfile;
}

}  // namespace data
}  // namespace mrmd