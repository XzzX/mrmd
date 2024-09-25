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
namespace util
{
void dumpView(const std::string& filename, const ScalarView& view)
{
    auto data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    std::ofstream fout(filename);
    for (idx_t i = 0; i < idx_c(data.extent(0)); ++i)
    {
        fout << data(i) << std::endl;
    }
    fout.close();
}

void dumpView(const std::string& filename, const MultiView& view)
{
    auto data = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    std::ofstream fout(filename);
    for (idx_t i = 0; i < idx_c(data.extent(0)); ++i)
    {
        for (idx_t j = 0; j < idx_c(data.extent(1)); ++j)
        {
            fout << data(i, j) << " ";
        }
        fout << std::endl;
    }
    fout.close();
}
}  // namespace util
}  // namespace mrmd
