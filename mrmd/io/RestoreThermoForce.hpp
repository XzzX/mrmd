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

#include "data/Subdomain.hpp"
#include "action/ThermodynamicForce.hpp"
#include <fstream>
#include "datatypes.hpp"

namespace mrmd
{
namespace io
{
    action::ThermodynamicForce restoreThermoForce(const std::string& filename, const data::Subdomain& subdomain, const real_t& targetDensity = 1_r, const real_t& thermodynamicForceModulation = 1_r, const bool enforceSymmetry = false, const bool usePeriodicity = false, const idx_t& maxGridSize = 10000, const idx_t& maxNumForces = 5)
    {
        std::string line;
        std::string word;
        int binNum = 0;
        int histNum = 0;
        ScalarView gridRead("grid", maxGridSize);
        MultiView forcesRead("thermodynamic-force", maxGridSize, maxNumForces);

        std::ifstream fileThermoForce(filename);        
        std::getline(fileThermoForce, line);
        std::stringstream gridLineStream(line);
        while (gridLineStream >> word)
        {
            gridRead(binNum) = std::stold(word);
            binNum++;
            MRMD_HOST_ASSERT_LESS(binNum, maxGridSize);
        }
        ScalarView grid = Kokkos::subview(gridRead, Kokkos::make_pair(0, binNum));
        real_t binWidth = grid(1) - grid(0);
        binNum = 0;

        //MRMD_HOST_ASSERT_EQUAL(grid(0) - subdomain.minCorner[0], subdomain.maxCorner[0] - grid(binNum));
        //MRMD_HOST_ASSERT_LESSEQUAL(grid(0) - subdomain.minCorner[0], grid(1) - grid(0))

        while (std::getline(fileThermoForce, line))
        {
            std::stringstream forceLineStream(line);
            while (forceLineStream >> word)
            {
                forcesRead(binNum, histNum) = std::stold(word);
                binNum++;
            }
            histNum++;

            MRMD_HOST_ASSERT_LESS(histNum, maxNumForces);
        }
        fileThermoForce.close();  
        
        MultiView forces = Kokkos::subview(forcesRead, Kokkos::make_pair(0, binNum), Kokkos::make_pair(0, histNum));
        
        //MRMD_HOST_ASSERT_EQUAL(grid, forcesHist.createGrid());

        action::ThermodynamicForce thermodynamicForce(targetDensity, subdomain, binWidth, thermodynamicForceModulation, enforceSymmetry, usePeriodicity);
        thermodynamicForce.setForce(forces);

        return thermodynamicForce;
    }
}  // namespace io
}  // namespace mrmd