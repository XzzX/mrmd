// Copyright 2025 Sebastian Eibl
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

#include "CountingPlane.hpp"

#include "assert/assert.hpp"
#include "util/Kokkos_grow.hpp"

namespace mrmd::analysis
{
CountingPlane::CountingPlane(const Point3D& pointOnPlane, Vector3D planeNormal)
    : pointOnPlane_(pointOnPlane), planeNormal_(planeNormal), distanceToPlane_("distanceToPlane", 0)
{
}

void CountingPlane::startCounting(data::Atoms& atoms)
{
    auto pointOnPlane = pointOnPlane_;
    auto planeNormal = planeNormal_;

    auto pos = atoms.getPos();
    util::grow(distanceToPlane_, atoms.size());
    auto distanceToPlane = distanceToPlane_;
    Kokkos::parallel_for(
        "ComputeDistanceToPlane",
        Kokkos::RangePolicy<>(0, atoms.size()),
        KOKKOS_LAMBDA(const idx_t i) {
            distanceToPlane(i) = (pos(i, 0) - pointOnPlane[0]) * planeNormal[0] +
                                 (pos(i, 1) - pointOnPlane[1]) * planeNormal[1] +
                                 (pos(i, 2) - pointOnPlane[2]) * planeNormal[2];
        });
}

int64_t CountingPlane::stopCounting(data::Atoms& atoms)
{
    auto pointOnPlane = pointOnPlane_;
    auto planeNormal = planeNormal_;

    auto pos = atoms.getPos();
    MRMD_HOST_CHECK_GREATEREQUAL(distanceToPlane_.size(),
                                 pos.size(),
                                 "You must call startCounting before stopCounting. The number of "
                                 "particles is not allowed to change!");
    auto distanceToPlane = distanceToPlane_;
    int64_t count = 0;
    Kokkos::parallel_reduce(
        "CountCrossings",
        Kokkos::RangePolicy<>(0, atoms.size()),
        KOKKOS_LAMBDA(const idx_t i, int64_t& localCount) {
            auto dist = (pos(i, 0) - pointOnPlane[0]) * planeNormal[0] +
                        (pos(i, 1) - pointOnPlane[1]) * planeNormal[1] +
                        (pos(i, 2) - pointOnPlane[2]) * planeNormal[2];
            if (dist * distanceToPlane(i) < 0)
            {
                localCount += 1;
            }
        },
        count);
    return count;
}

}  // namespace mrmd::analysis