// Copyright 2024 Sebastian Eibl
// Copyright 2026 Julian Friedrich Hille
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

#include "AxialAverageProfile.hpp"

namespace mrmd
{
namespace analysis
{
AxialAverageProfile::AxialAverageProfile(const data::Subdomain& subdomain,
                                         const real_t binWidth,
                                         const real_t normalizationFactor,
                                         const idx_t numTypes,
                                         const AXIS& axis)
    : averageProfile_("average-profile",
                      subdomain.minCorner[to_underlying(axis)],
                      subdomain.maxCorner[to_underlying(axis)],
                      idx_c(subdomain.diameter[to_underlying(axis)] / binWidth),
                      numTypes),
      normalizationFactor_(normalizationFactor),
      numTypes_(numTypes),
      axis_(axis)
{
    MRMD_HOST_CHECK_FLOAT_EQUAL(
        averageProfile_.binSize, binWidth, "requested bin size is not achieved");

    MRMD_HOST_CHECK_GREATER(numTypes, 0);
}

void AxialAverageProfile::reset()
{
    MRMD_HOST_CHECK_GREATER(
        numberOfSamples_,
        0,
        "Cannot reset AxialAverageProfile because no samples have been taken yet.");

    Kokkos::deep_copy(averageProfile_.data, 0_r);
    numberOfSamples_ = 0;
}

}  // namespace analysis
}  // namespace mrmd