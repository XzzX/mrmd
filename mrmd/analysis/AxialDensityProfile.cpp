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

#include "AxialDensityProfile.hpp"

#include <fmt/format.h>

namespace mrmd
{
namespace analysis
{

data::MultiHistogram getAxialDensityProfile(const idx_t numAtoms,
                                            const data::Atoms::pos_t& positions,
                                            const data::Atoms::type_t& types,
                                            const int64_t numTypes,
                                            const real_t min,
                                            const real_t max,
                                            const int64_t numBins,
                                            const real_t binWidth,
                                            const int64_t axis)
{
    MRMD_HOST_CHECK_GREATEREQUAL(max, min);
    MRMD_HOST_CHECK_GREATER(numTypes, 0);
    MRMD_HOST_CHECK_GREATEREQUAL(axis, 0);
    MRMD_HOST_CHECK_LESSEQUAL(axis, 3);

    data::MultiHistogram histogram("density-profile", min, max, numBins, numTypes);
    MultiScatterView scatter(histogram.data);

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numAtoms, numBins});
    auto kernel = KOKKOS_LAMBDA(const idx_t atomIdx, const idx_t binIdx)
    {
        MRMD_DEVICE_ASSERT_GREATEREQUAL(types(atomIdx), 0);
        MRMD_DEVICE_ASSERT_LESS(types(atomIdx), numTypes);
        auto binPos = histogram.getBinPosition(binIdx);
        auto atomPos = positions(atomIdx, axis);
        auto access = scatter.access();
        if (atomPos >= binPos - binWidth / 2 && atomPos <= binPos + binWidth / 2)
        {
            access(binIdx, types(atomIdx)) += 1_r;
        }
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::Experimental::contribute(histogram.data, scatter);
    Kokkos::fence();

    return histogram;
}

}  // namespace analysis
}  // namespace mrmd