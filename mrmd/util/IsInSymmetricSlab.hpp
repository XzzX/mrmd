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
class IsInSymmetricSlab
{
private:
    const Point3D center_;
    const real_t slabMin_;
    const real_t slabMax_;

public:
    KOKKOS_INLINE_FUNCTION
    bool operator()(const real_t& x, const real_t& /*y*/, const real_t& /*z*/) const
    {
        auto dx = x - center_[0];
        auto absDx = std::abs(dx);

        return (absDx >= slabMin_ && absDx <= slabMax_);
    }

    IsInSymmetricSlab(const Point3D& center, const real_t slabMin, const real_t slabMax)
        : center_(center), slabMin_(slabMin), slabMax_(slabMax)
    {
    }
};
}  // namespace util
}  // namespace mrmd