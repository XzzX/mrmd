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

#include "io/DumpGRO.hpp"

#include <fstream>

namespace mrmd
{
namespace io
{
/// format specification
/// https://manual.gromacs.org/current/reference-manual/file-formats.html?highlight=gro#gro
void dumpGRO(const std::string& filename,
             data::Atoms& atoms,
             const data::Subdomain& subdomain,
             const real_t& timestamp,
             const std::string& title,
             bool dumpGhosts,
             bool dumpVelocities,
             const std::string& resName,
             std::vector<std::string> typeNames)
{
    // very ugly, will also copy the whole atom data which is unnecessary, custom slicing
    // required
    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto typ = Cabana::slice<data::Atoms::TYPE>(hAoSoA);

    std::ofstream fout(filename);
    if (!fout.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    auto lastAtomIdx = atoms.numLocalAtoms + (dumpGhosts ? atoms.numGhostAtoms : 0);
    fout << title << ", t=" << timestamp << std::endl;
    fout << lastAtomIdx << std::endl;

    for (idx_t idx = 0; idx < lastAtomIdx; ++idx)
    {
        std::string typeName = typeNames[typ(idx)];

        char buf[1024];
        if (!dumpVelocities)
        {
            sprintf(buf,
                    "%5d%-5s%5s%5d%8.3f%8.3f%8.3f",
                    int_c(idx + 1),
                    resName.c_str(),
                    typeName.c_str(),
                    int_c(idx + 1),
                    pos(idx, 0),
                    pos(idx, 1),
                    pos(idx, 2));
        }
        else
        {
            sprintf(buf,
                    "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f",
                    int_c(idx + 1),
                    resName.c_str(),
                    typeName.c_str(),
                    int_c(idx + 1),
                    pos(idx, 0),
                    pos(idx, 1),
                    pos(idx, 2),
                    vel(idx, 0),
                    vel(idx, 1),
                    vel(idx, 2));
        }
        fout << std::string(buf) << std::endl;
    }
    fout << "    " << subdomain.diameter[0] << " " << subdomain.diameter[1] << " "
         << subdomain.diameter[2] << std::endl;
    fout.close();
}
}  // namespace io
}  // namespace mrmd