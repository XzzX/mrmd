#include "Subdomain.hpp"

#include "constants.hpp"

namespace mrmd
{
namespace data
{
void Subdomain::scale(const real_t& scalingFactor)
{
    for (auto dim = 0; dim < DIMENSIONS; ++dim)
    {
        minCorner[dim] *= scalingFactor;
        maxCorner[dim] *= scalingFactor;
        minGhostCorner[dim] *= scalingFactor;
        maxGhostCorner[dim] *= scalingFactor;
        minInnerCorner[dim] *= scalingFactor;
        maxInnerCorner[dim] *= scalingFactor;
        diameter[dim] *= scalingFactor;
    }
    ghostLayerThickness *= scalingFactor;

    checkInvariants(*this);
}

void checkInvariants(const Subdomain& subdomain)
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