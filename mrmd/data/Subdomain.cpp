#include "Subdomain.hpp"

#include "constants.hpp"

namespace mrmd
{
namespace data
{
void Subdomain::scaleDim(const real_t& scalingFactor, const idx_t& dim)
{
    auto newMinCorner = minCorner;
    auto newMaxCorner = maxCorner;
    newMinCorner[dim] *= scalingFactor;
    newMaxCorner[dim] *= scalingFactor;
    *this = Subdomain(newMinCorner, newMaxCorner, ghostLayerThickness);
    checkInvariants(*this);
}

void Subdomain::scale(const real_t& scalingFactor)
{
    scaleDim(scalingFactor, COORD_X);
    scaleDim(scalingFactor, COORD_Y);
    scaleDim(scalingFactor, COORD_Z);
}

void checkInvariants([[maybe_unused]] const Subdomain& subdomain)
{
    for (auto dim = 0; dim < DIMENSIONS; ++dim)
    {
        assert(subdomain.minGhostCorner[dim] < subdomain.minCorner[dim]);
        assert(subdomain.minCorner[dim] < subdomain.minInnerCorner[dim]);
        assert(subdomain.minInnerCorner[dim] < subdomain.maxInnerCorner[dim]);
        assert(subdomain.maxInnerCorner[dim] < subdomain.maxCorner[dim]);
        assert(subdomain.maxCorner[dim] < subdomain.maxGhostCorner[dim]);

        assert(subdomain.diameter[dim] >= 0_r);
        assert(subdomain.diameter[dim] > subdomain.ghostLayerThickness &&
               "ghost layer to larger than subdomain");
    }
}

}  // namespace data
}  // namespace mrmd