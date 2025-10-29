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

/**
 * @brief Predicate to check if a point is inside a slab region along a given axis.
 * 
 * @tparam axis The axis along which the slab is defined, the slab is infinite in the other two axes.
 */
template <AXIS axis>
class IsInSlab
{
private:
    const real_t min_;
    const real_t max_;

public:

    KOKKOS_INLINE_FUNCTION
    bool operator()(const real_t& x, const real_t& y, const real_t& z) const
    {
        if constexpr (axis == COORD::X)
        {
            return (x >= min_ && x < max_);
        }
        else if constexpr (axis == COORD::Y)
        {
            return (y >= min_ && y < max_);
        }
        else if constexpr (axis == COORD::Z)
        {
            return (z >= min_ && z < max_);
        }
        else
        {
            static_assert(false, "Ifs need to be exhaustive");
            return false; // to suppress compiler warning
        }
    }

    IsInSlab(const real_t min,
             const real_t max)
        : min_(min),
          max_(max)
    {
    }
};

}  // namespace util
}  // namespace mrmd