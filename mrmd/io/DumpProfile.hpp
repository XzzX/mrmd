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

#include <fstream>

#include "datatypes.hpp"

namespace mrmd
{
namespace io
{
class DumpProfile
{
public:
    void open(const std::string& filename);

    void close();

    void dumpGrid(const ScalarView::HostMirror& grid);

    void dumpStep(const ScalarView::HostMirror& dataProfile,
                  const real_t& normalizationFactor = 1_r);

private:
    std::ofstream fileProfile_;
};

void dumpSingleProfile(const std::string& filename,
                       const ScalarView::HostMirror& grid,
                       const ScalarView::HostMirror& dataProfile,
                       const real_t& normalizationFactor = 1_r);
}  // namespace io
}  // namespace mrmd