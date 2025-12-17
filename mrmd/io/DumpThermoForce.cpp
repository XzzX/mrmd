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

#include "DumpThermoForce.hpp"

#include "DumpProfile.hpp"

namespace mrmd
{
namespace io
{
void dumpThermoForce(const std::string& filename,
                     const action::ThermodynamicForce& thermodynamicForce,
                     const idx_t& typeId)
{
    ScalarView::HostMirror grid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), thermodynamicForce.getForce().createGrid());
    auto numBins = grid.size();
    ScalarView::HostMirror forceView("forceView", numBins);
    auto thermoForce = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           thermodynamicForce.getForce(typeId));
    for (size_t idx = 0; idx < numBins; ++idx)
    {
        forceView(idx) = thermoForce(idx);
    }

    dumpSingleProfile(filename, grid, forceView);
}

void dumpThermoForce(const std::string& filename,
                     const action::ThermodynamicForce& thermodynamicForce)
{
    DumpProfile dumpThermoForce;
    ScalarView::HostMirror grid = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), thermodynamicForce.getForce().createGrid());
    auto numBins = grid.size();

    dumpThermoForce.open(filename);
    dumpThermoForce.dumpScalarView(grid);
    for (idx_t typeId = 0; typeId < thermodynamicForce.getForce().numHistograms; typeId++)
    {
        ScalarView::HostMirror forceView("forceTest", numBins);
        auto thermoForce = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                               thermodynamicForce.getForce(typeId));
        for (size_t idx = 0; idx < numBins; ++idx)
        {
            forceView(idx) = thermoForce(idx);
        }
        dumpThermoForce.dumpScalarView(forceView);
    }
    dumpThermoForce.close();
}
}  // namespace io
}  // namespace mrmd