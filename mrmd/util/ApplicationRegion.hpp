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

#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
class ApplicationRegion
{
    private:
    const Point3D center_;
    const real_t applicationRegionMin_;
    const real_t applicationRegionMax_;

public:
    KOKKOS_INLINE_FUNCTION
    bool isInApplicationRegion(const real_t& x, const real_t& /*y*/, const real_t& /*z*/) const
    {
        auto dx = x - center_[0];
        auto absDx = std::abs(dx);

        return (absDx >= applicationRegionMin_ && absDx <= applicationRegionMax_);
    }

    ApplicationRegion(const Point3D& center,
         const real_t applicationRegionMin,
         const real_t applicationRegionMax)
        : center_(center),
          applicationRegionMin_(applicationRegionMin),
          applicationRegionMax_(applicationRegionMax)
    {
    }
};
} // namespace util
} // namespace mrmd