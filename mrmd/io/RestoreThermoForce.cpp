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

#include "RestoreThermoForce.hpp"

#include <fstream>

namespace mrmd
{
namespace io
{
action::ThermodynamicForce restoreThermoForce(
    const std::string& filename,
    const data::Subdomain& subdomain,
    const std::vector<real_t>& targetDensities,
    const std::vector<real_t>& thermodynamicForceModulations,
    const bool enforceSymmetry,
    const bool usePeriodicity,
    const idx_t maxNumForces,
    const real_t requestedDensityBinWidth)
{
    std::string line;
    std::string word;
    int binNumForce = 0;
    int histNum = 0;

    std::ifstream fileThermoForce(filename);
    std::getline(fileThermoForce, line);
    std::stringstream gridLineStream(line);
    while (gridLineStream >> word)
    {
        binNumForce++;
    }
    MRMD_HOST_ASSERT_GREATER(binNumForce, 1);

    MultiView::HostMirror h_forcesRead("h_forcesRead", binNumForce, maxNumForces);

    while (std::getline(fileThermoForce, line))
    {
        binNumForce = 0;
        std::stringstream forceLineStream(line);
        while (forceLineStream >> word)
        {
            h_forcesRead(binNumForce, histNum) = std::stod(word);
            binNumForce++;
        }
        histNum++;

        MRMD_HOST_ASSERT_LESSEQUAL(histNum, maxNumForces);
    }
    fileThermoForce.close();

    auto h_forces = Kokkos::subview(
        h_forcesRead, Kokkos::make_pair(0, binNumForce), Kokkos::make_pair(0, histNum));
    MultiView d_forces("d_forces", binNumForce, histNum);
    Kokkos::deep_copy(d_forces, h_forces);

    auto forceBinNumber = idx_c(binNumForce);
    real_t densityBinWidth = requestedDensityBinWidth;

    if (requestedDensityBinWidth < 0_r)
    {
        densityBinWidth = (subdomain.diameter[0] / real_c(forceBinNumber));
    }

    action::ThermodynamicForce thermodynamicForce(targetDensities,
                                                  subdomain,
                                                  forceBinNumber,
                                                  densityBinWidth,
                                                  thermodynamicForceModulations,
                                                  enforceSymmetry,
                                                  usePeriodicity);

    thermodynamicForce.setForce(d_forces);

    return thermodynamicForce;
}
}  // namespace io
}  // namespace mrmd