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
class ExponentialMovingAverage
{
private:
    real_t alpha_;  ///< weighting factor
    real_t beta_;   ///< 1-alpha

    real_t val_ = 0_r;
    bool isFirstVal_ = true;

public:
    explicit ExponentialMovingAverage(const real_t& weightingFactor)
        : alpha_(weightingFactor), beta_(1_r - weightingFactor)
    {
    }

    inline operator real_t() const { return toReal(); }
    inline real_t toReal() const { return val_; }

    void append(const real_t& val)
    {
        if (isFirstVal_)
        {
            isFirstVal_ = false;
            val_ = val;
            return;
        }
        val_ = val * alpha_ + val_ * beta_;
    }
};

inline ExponentialMovingAverage& operator<<(ExponentialMovingAverage& lhs, const real_t& rhs)
{
    lhs.append(rhs);
    return lhs;
}

}  // namespace util
}  // namespace mrmd
