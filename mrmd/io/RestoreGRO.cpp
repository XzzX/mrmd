// Copyright 2024 Sebastian Eibl
// Copyright 2026 Julian Friedrich Hille
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

#include "RestoreGRO.hpp"

#include <cstdio>
#include <fstream>
#include <iostream>

namespace mrmd::io
{
void restoreGRO(const std::string& filename,
                data::Subdomain& subdomain,
                data::Atoms& atoms,
                const bool& containsGhostAtoms,
                const bool& containsVelocities)
{
    int tmpInt;
    char tmpChar[6];
    double tmpFloat[6];

    std::ifstream fin(filename);
    if (!fin.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    if (containsGhostAtoms)
    {
        std::cout << "Warning: The file " << filename
                  << " contains ghost atoms. This is not supported by restoreGRO and will likely "
                     "lead to unexpected behavior."
                  << std::endl;

        return;
    }
    if (!containsVelocities)
    {
        std::cout << "Warning: The file " << filename
                  << " does not contain velocities. This is not supported by restoreGRO and will "
                     "likely lead to unexpected behavior."
                  << std::endl;

        return;
    }
    char buf[1024];

    // Skip the first line of the file which contains a comment.
    fin.getline(buf, 1024);

    fin.getline(buf, 1024);
    sscanf(buf, "%d", &tmpInt);
    idx_t numAtoms = idx_c(tmpInt);
    assert(numAtoms > 0);
    atoms.resize(numAtoms);

    auto d_Atoms = atoms.getAoSoA();
    auto h_Atoms = Cabana::create_mirror_view(Kokkos::HostSpace(), d_Atoms);
    auto h_pos = Cabana::slice<data::Atoms::POS>(h_Atoms);
    auto h_vel = Cabana::slice<data::Atoms::VEL>(h_Atoms);
    auto h_mass = Cabana::slice<data::Atoms::MASS>(h_Atoms);
    auto h_relativeMass = Cabana::slice<data::Atoms::RELATIVE_MASS>(h_Atoms);

    idx_t idx = 0;
    while (idx < numAtoms && !fin.eof())
    {
        fin.getline(buf, 1024);
        sscanf(buf,
               "%5d%5s%5s%5d%8lf%8lf%8lf%8lf%8lf%8lf",
               &tmpInt,
               tmpChar,
               tmpChar,
               &tmpInt,
               &tmpFloat[0],
               &tmpFloat[1],
               &tmpFloat[2],
               &tmpFloat[3],
               &tmpFloat[4],
               &tmpFloat[5]);

        h_pos(idx, 0) = real_c(tmpFloat[0]);
        h_pos(idx, 1) = real_c(tmpFloat[1]);
        h_pos(idx, 2) = real_c(tmpFloat[2]);
        h_vel(idx, 0) = real_c(tmpFloat[3]);
        h_vel(idx, 1) = real_c(tmpFloat[4]);
        h_vel(idx, 2) = real_c(tmpFloat[5]);

        h_mass(idx) = 1_r;

        h_relativeMass(idx) = 1_r;

        ++idx;
    }

    fin >> subdomain.diameter[0] >> subdomain.diameter[1] >> subdomain.diameter[2];

    data::Subdomain tmpSubdomain(subdomain.minCorner,
                                 {subdomain.minCorner[0] + subdomain.diameter[0],
                                  subdomain.minCorner[1] + subdomain.diameter[1],
                                  subdomain.minCorner[2] + subdomain.diameter[2]},
                                 subdomain.ghostLayerThickness);

    subdomain = tmpSubdomain;

    fin.close();

    Cabana::deep_copy(d_Atoms, h_Atoms);
    assert(idx == numAtoms);
    atoms.numLocalAtoms = idx;
    atoms.numGhostAtoms = 0;

    auto force = atoms.getForce();
    Cabana::deep_copy(force, 0_r);
    auto type = atoms.getType();
    Cabana::deep_copy(type, 0);
}
}  // namespace mrmd::io