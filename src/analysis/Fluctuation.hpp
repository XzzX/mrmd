#pragma once

#include "data/MultiHistogram.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
real_t getFluctuation(const data::MultiHistogram& hist,
                      const real_t& reference,
                      const idx_t& specimen);

}  // namespace analysis
}  // namespace mrmd