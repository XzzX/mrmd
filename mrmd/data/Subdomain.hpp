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

    Subdomain(const Point3D& minCornerArg,
              const Point3D& maxCornerArg,
              const Vector3D& ghostLayerThicknessArg)
        : minCorner(minCornerArg),
          maxCorner(maxCornerArg),
          ghostLayerThickness(ghostLayerThicknessArg)
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            MRMD_HOST_CHECK_GREATEREQUAL(ghostLayerThicknessArg[dim], 0_r);

            minGhostCorner[dim] = minCorner[dim] - ghostLayerThickness[dim];
            maxGhostCorner[dim] = maxCorner[dim] + ghostLayerThickness[dim];

            minInnerCorner[dim] = minCorner[dim] + ghostLayerThickness[dim];
            maxInnerCorner[dim] = maxCorner[dim] - ghostLayerThickness[dim];

            diameter[dim] = maxCorner[dim] - minCorner[dim];
            diameterWithGhostLayer[dim] =
                maxCorner[dim] - minCorner[dim] + 2_r * ghostLayerThickness[dim];
        }
    }
    Subdomain(const Point3D& minCornerArg,
              const Point3D& maxCornerArg,
              real_t ghostLayerThicknessArg)
        : Subdomain(
              minCornerArg,
              maxCornerArg,
              Vector3D{ghostLayerThicknessArg, ghostLayerThicknessArg, ghostLayerThicknessArg})
    {
    }

    void scaleDim(const real_t& scalingFactor, const AXIS& axis);
    void scale(const real_t& scalingFactor);

    Point3D minCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                         std::numeric_limits<real_t>::signaling_NaN(),
                         std::numeric_limits<real_t>::signaling_NaN()};
    Point3D maxCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                         std::numeric_limits<real_t>::signaling_NaN(),
                         std::numeric_limits<real_t>::signaling_NaN()};  // namespace data

    Point3D ghostLayerThickness = {std::numeric_limits<real_t>::signaling_NaN(),
                                   std::numeric_limits<real_t>::signaling_NaN(),
                                   std::numeric_limits<real_t>::signaling_NaN()};

    Point3D minGhostCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                              std::numeric_limits<real_t>::signaling_NaN(),
                              std::numeric_limits<real_t>::signaling_NaN()};
    Point3D maxGhostCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                              std::numeric_limits<real_t>::signaling_NaN(),
                              std::numeric_limits<real_t>::signaling_NaN()};

    Point3D minInnerCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                              std::numeric_limits<real_t>::signaling_NaN(),
                              std::numeric_limits<real_t>::signaling_NaN()};
    Point3D maxInnerCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                              std::numeric_limits<real_t>::signaling_NaN(),
                              std::numeric_limits<real_t>::signaling_NaN()};

    Vector3D diameter = {std::numeric_limits<real_t>::signaling_NaN(),
                         std::numeric_limits<real_t>::signaling_NaN(),
                         std::numeric_limits<real_t>::signaling_NaN()};

    Vector3D diameterWithGhostLayer = {std::numeric_limits<real_t>::signaling_NaN(),
                                       std::numeric_limits<real_t>::signaling_NaN(),
                                       std::numeric_limits<real_t>::signaling_NaN()};
};

void checkInvariants(const Subdomain& subdomain);

}  // namespace data
}  // namespace mrmd