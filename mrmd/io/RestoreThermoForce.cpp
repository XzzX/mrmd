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
    const idx_t maxNumForces)
{
    std::string line;
    std::string word;
    int binNum = 0;
    int histNum = 0;
    real_t grid0;
    real_t grid1;

    std::ifstream fileThermoForce(filename);
    std::getline(fileThermoForce, line);
    std::stringstream gridLineStream(line);
    while (gridLineStream >> word)
    {
        if (binNum == 0)
        {
            grid0 = std::stod(word);
        }
        if (binNum == 1)
        {
            grid1 = std::stod(word);
        }
        binNum++;
    }
    real_t binWidth = grid1 - grid0;

    MultiView::HostMirror h_forcesRead("h_forcesRead", binNum, maxNumForces);

    while (std::getline(fileThermoForce, line))
    {
        binNum = 0;
        std::stringstream forceLineStream(line);
        while (forceLineStream >> word)
        {
            h_forcesRead(binNum, histNum) = std::stod(word);
            binNum++;
        }
        histNum++;

        MRMD_HOST_ASSERT_LESSEQUAL(histNum, maxNumForces);
    }
    fileThermoForce.close();

    auto h_forces =
        Kokkos::subview(h_forcesRead, Kokkos::make_pair(0, binNum), Kokkos::make_pair(0, histNum));
    MultiView d_forces("d_forces", binNum, histNum);
    Kokkos::deep_copy(d_forces, h_forces);

    action::ThermodynamicForce thermodynamicForce(targetDensities,
                                                  subdomain,
                                                  binWidth,
                                                  thermodynamicForceModulations,
                                                  enforceSymmetry,
                                                  usePeriodicity);

    thermodynamicForce.setForce(d_forces);

    return thermodynamicForce;
}
}  // namespace io
}  // namespace mrmd