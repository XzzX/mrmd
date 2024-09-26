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

#include "io/DumpCSV.hpp"

#include <fstream>
#include <iostream>

#include "data/Atoms.hpp"

namespace mrmd
{
namespace io
{
void dumpCSV(const std::string& filename, data::Atoms& atoms, bool dumpGhosts)
{
    data::HostAtoms at(atoms);
    auto pos = at.getPos();
    auto vel = at.getVel();
    auto type = at.getType();

    std::ofstream fout(filename);
    if (!fout.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    fout << "idx, mol, type, ghost, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z" << std::endl;
    auto lastAtomIdx = at.numLocalAtoms + (dumpGhosts ? at.numGhostAtoms : 0);
    for (idx_t idx = 0; idx < lastAtomIdx; ++idx)
    {
        fout << idx << ", " << idx / 3 << ", " << type(idx) << ", "
             << ((idx < at.numLocalAtoms) ? 0 : 1) << ", " << pos(idx, 0) << ", " << pos(idx, 1)
             << ", " << pos(idx, 2) << ", " << vel(idx, 0) << ", " << vel(idx, 1) << ", "
             << vel(idx, 2) << std::endl;
    }
    fout.close();
}
}  // namespace io
}  // namespace mrmd
