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

#include <array>
#include <cassert>

#include "assert/assert.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
/**
 * |X|O|OOOOOOOOOO
 * |X|O|OOOOOOOOOO
 * |X|O+----------  <-- minInnerCorner
 * |X|OOOOOOOOOOOO
 * |X+------------  <-- minCorner
 * |XXXXXXXXXXXXXX
 * +--------------  <-- minGhostCorner
 */
struct Subdomain
{
    Subdomain() = default;
    Subdomain(const std::array<real_t, 3>& minCornerArg,
              const std::array<real_t, 3>& maxCornerArg,
              real_t ghostLayerThicknessArg)
        : minCorner(minCornerArg),
          maxCorner(maxCornerArg),
          ghostLayerThickness(ghostLayerThicknessArg)
    {
        MRMD_HOST_CHECK_GREATEREQUAL(ghostLayerThicknessArg, 0_r);

        for (auto dim = 0; dim < 3; ++dim)
        {
            minGhostCorner[dim] = minCorner[dim] - ghostLayerThickness;
            maxGhostCorner[dim] = maxCorner[dim] + ghostLayerThickness;

            minInnerCorner[dim] = minCorner[dim] + ghostLayerThickness;
            maxInnerCorner[dim] = maxCorner[dim] - ghostLayerThickness;

            diameter[dim] = maxCorner[dim] - minCorner[dim];
            diameterWithGhostLayer[dim] =
                maxCorner[dim] - minCorner[dim] + 2_r * ghostLayerThickness;
        }
    }

    void scaleDim(const real_t& scalingFactor, const idx_t& dim);
    void scale(const real_t& scalingFactor);

    std::array<real_t, 3> minCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                       std::numeric_limits<real_t>::signaling_NaN(),
                                       std::numeric_limits<real_t>::signaling_NaN()};
    std::array<real_t, 3> maxCorner = {
        std::numeric_limits<real_t>::signaling_NaN(),
        std::numeric_limits<real_t>::signaling_NaN(),
        std::numeric_limits<real_t>::signaling_NaN()};  // namespace data

    real_t ghostLayerThickness = std::numeric_limits<real_t>::signaling_NaN();

    std::array<real_t, 3> minGhostCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN()};
    std::array<real_t, 3> maxGhostCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN()};

    std::array<real_t, 3> minInnerCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN()};
    std::array<real_t, 3> maxInnerCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN()};

    std::array<real_t, 3> diameter = {std::numeric_limits<real_t>::signaling_NaN(),
                                      std::numeric_limits<real_t>::signaling_NaN(),
                                      std::numeric_limits<real_t>::signaling_NaN()};

    std::array<real_t, 3> diameterWithGhostLayer = {std::numeric_limits<real_t>::signaling_NaN(),
                                                    std::numeric_limits<real_t>::signaling_NaN(),
                                                    std::numeric_limits<real_t>::signaling_NaN()};
};

void checkInvariants(const Subdomain& subdomain);

}  // namespace data
}  // namespace mrmd