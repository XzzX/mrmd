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

#include "action/ThermodynamicForce.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace io
{
action::ThermodynamicForce restoreThermoForce(
    const std::string& filename,
    const data::Subdomain& subdomain,
    const std::vector<real_t>& targetDensities = {1_r},
    const std::vector<real_t>& thermodynamicForceModulations = {1_r},
    const bool enforceSymmetry = false,
    const bool usePeriodicity = false,
    const idx_t maxNumForces = 10);
}  // namespace io
}  // namespace mrmd