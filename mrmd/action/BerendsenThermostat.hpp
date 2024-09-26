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

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
namespace BerendsenThermostat
{
/**
 * Berendsen Thermostat
 * DOI: 10.1063/1.448118
 */
void apply(data::Atoms& atoms,
           const real_t& currentTemperature,
           const real_t& targetTemperature,
           const real_t& gamma);
}  // namespace BerendsenThermostat
}  // namespace action
}  // namespace mrmd