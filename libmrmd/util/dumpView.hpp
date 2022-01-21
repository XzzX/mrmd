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
