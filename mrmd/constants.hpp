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
/// number of spatial dimensions
constexpr static int DIMENSIONS = 3;

constexpr real_t pi = 3.14159265358979323846;
constexpr real_t M_SQRTPI = 1.77245385090551602729;  // sqrt(pi)

namespace toSI
{
constexpr real_t temperature = 1e3_r / (6.02214129_r * 1.380649_r);
constexpr real_t energy = 1e3_r / 6.02214129e23_r;
}  // namespace toSI
}  // namespace mrmd