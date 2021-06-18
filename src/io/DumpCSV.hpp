#pragma once

#include <fstream>

#include "Particles.hpp"

inline void dumpCSV(const std::string& filename, Particles& particles)
{
    auto pos = particles.getPos();
    auto vel = particles.getVel();
    auto ghost = particles.getGhost();

    std::ofstream fout(filename);
    fout << "pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, is_ghost" << std::endl;
    for (idx_t idx = 0; idx < particles.numLocalParticles + particles.numGhostParticles; ++idx)
    {
        fout << pos(idx, 0) << ", " << pos(idx, 1) << ", " << pos(idx, 2) << ", " << vel(idx, 0)
             << ", " << vel(idx, 1) << ", " << vel(idx, 2) << ", " << ((ghost(idx) == -1) ? 0 : 1)
             << std::endl;
    }
    fout.close();
}
