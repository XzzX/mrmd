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

#include <string>

#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
struct Histogram
{
    Histogram(const std::string& label, const real_t minArg, const real_t maxArg, idx_t numBinsArg)
        : min(minArg),
          max(maxArg),
          numBins(numBinsArg),
          binSize((max - min) / real_c(numBins)),
          inverseBinSize(1_r / binSize),
          data(label, numBins)
    {
    }

    /**
     * @param val input value
     * @return corresponding bin or -1 if outside of range
     */
    KOKKOS_INLINE_FUNCTION idx_t getBin(const real_t& val) const
    {
        auto bin = idx_c((val - min) * inverseBinSize);
        if (bin < 0) bin = -1;
        if (bin >= numBins) bin = -1;
        return bin;
    }

    const real_t min;
    const real_t max;
    const idx_t numBins;
    const real_t binSize;
    const real_t inverseBinSize;
    ScalarView data;

    Histogram& operator+=(const Histogram& rhs);
};

data::Histogram gradient(const data::Histogram& input);

/**
 * Overload for writing to console/files.
 *
 * \attention If the data is located in device memory the data will be copied into host memory.
 */
std::ostream& operator<<(std::ostream& os, const data::Histogram& hist);

}  // namespace data
}  // namespace mrmd