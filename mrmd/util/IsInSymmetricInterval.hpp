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

#pragma once

#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
class IsInSymmetricInterval
{
private:
    const real_t center_;
    const real_t intervalMin_;
    const real_t intervalMax_;
    const real_t tolerance_;

public:
    KOKKOS_INLINE_FUNCTION
    bool operator()(const real_t& x) const
    {
        auto dx = x - center_;
        auto absDx = std::abs(dx);

        return (absDx >= intervalMin_ - tolerance_ && absDx <= intervalMax_ + tolerance_);
    }

    IsInSymmetricInterval(const real_t center,
                          const real_t intervalMin,
                          const real_t intervalMax,
                          const real_t tolerance = 0_r)
        : center_(center),
          intervalMin_(intervalMin),
          intervalMax_(intervalMax),
          tolerance_(tolerance)
    {
    }
};
}  // namespace util
}  // namespace mrmd