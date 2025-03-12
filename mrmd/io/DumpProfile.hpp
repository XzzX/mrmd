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
    void open(const std::string& filename, const ScalarView& grid)
    {
        fileProfile_.open(filename);

        for (auto i = 0; i < grid.extent(0); ++i)
        {
            std::string separator = (i < grid.extent(0) - 1) ? " " : "";
            fileProfile_ << grid(i) << separator;
        }
        fileProfile_ << std::endl;    
    }

    void close()
    {
        fileProfile_.close();
    }

    void dumpStep(const ScalarView& dataProfile, const real_t& normalizationFactor = 1_r)
    {
        for (auto i = 0; i < dataProfile.extent(0); ++i)
        {
            std::string separator = (i < dataProfile.extent(0) - 1) ? " " : "";
            fileProfile_ << dataProfile(i) * normalizationFactor << separator;
        }
        fileProfile_ << std::endl;    
    }

    void dump(const std::string& filename, const ScalarView& grid, const ScalarView& dataProfile, const real_t& normalizationFactor = 1_r)
    {
        open(filename, grid);
        dumpStep(dataProfile, normalizationFactor);
        close();
    }
private:
    std::ofstream fileProfile_;
};
}  // namespace io
}  // namespace mrmd