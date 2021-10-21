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

    operator real_t() const { return val_; }

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
