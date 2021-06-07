#include "Particles.hpp"

void Particles::copy(const idx_t src, const idx_t dst)
{
    for (auto dim = 0; dim < 3; ++dim)
    {
        getPos()(dst, dim) = getPos()(src, dim);
        getVel()(dst, dim) = getVel()(src, dim);
    }
}