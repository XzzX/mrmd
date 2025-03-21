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

#include "DumpProfile.hpp"

namespace mrmd
{
namespace io
{
void DumpProfile::open(const std::string& filename) { fileProfile_.open(filename); }

void DumpProfile::close() { fileProfile_.close(); }

void DumpProfile::dumpGrid(const ScalarView::HostMirror& grid) { dumpStep(grid); }

void DumpProfile::dumpStep(const ScalarView::HostMirror& dataProfile,
                           const real_t& normalizationFactor)
{
    for (size_t idx = 0; idx < dataProfile.extent(0); ++idx)
    {
        std::string separator = (idx < dataProfile.extent(0) - 1) ? " " : "";
        fileProfile_ << dataProfile(idx) * normalizationFactor << separator;
    }
    fileProfile_ << std::endl;
}

void dumpSingleProfile(const std::string& filename,
                       const ScalarView::HostMirror& grid,
                       const ScalarView::HostMirror& dataProfile,
                       const real_t& normalizationFactor)
{
    DumpProfile dumpProfile;
    dumpProfile.open(filename);
    dumpProfile.dumpGrid(grid);
    dumpProfile.dumpStep(dataProfile, normalizationFactor);
    dumpProfile.close();
}
}  // namespace io
}  // namespace mrmd