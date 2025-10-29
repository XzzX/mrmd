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

#include <vector>

#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
/**
 * Calculate a discretized density profile along the x axis.
 * Out-of-bounds values are discarded.
 */
data::MultiHistogram getAxialDensityProfile(const idx_t numAtoms,
                                            const data::Atoms::pos_t& positions,
                                            const data::Atoms::type_t& types,
                                            const int64_t numTypes,
                                            const real_t min,
                                            const real_t max,
                                            const int64_t numBins,
                                            const AXIS axis);

}  // namespace analysis
}  // namespace mrmd