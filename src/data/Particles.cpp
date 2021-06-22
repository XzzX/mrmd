#include "Particles.hpp"

void Particles::copy(const idx_t dst, const idx_t src)
{
    for (auto dim = 0; dim < 3; ++dim)
    {
        pos(dst, dim) = pos(src, dim);
        vel(dst, dim) = vel(src, dim);
    }
}