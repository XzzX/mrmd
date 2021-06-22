#include "Particles.hpp"

void Particles::copy(const idx_t dst, const idx_t src)
{
    for (auto dim = 0; dim < 3; ++dim)
    {
        getPos()(dst, dim) = getPos()(src, dim);
        getVel()(dst, dim) = getVel()(src, dim);
    }
}