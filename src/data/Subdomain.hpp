#pragma once

#include <array>
#include <cassert>

#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
struct Subdomain
{
    Subdomain(const std::array<real_t, 3>& minCorner,
              const std::array<real_t, 3>& maxCorner,
              real_t ghostLayerThickness)
        : minCorner(minCorner), maxCorner(maxCorner), ghostLayerThickness(ghostLayerThickness)
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            minGhostCorner[dim] = minCorner[dim] - ghostLayerThickness;
            maxGhostCorner[dim] = maxCorner[dim] + ghostLayerThickness;

            minInnerCorner[dim] = minCorner[dim] + ghostLayerThickness;
            maxInnerCorner[dim] = maxCorner[dim] - ghostLayerThickness;

            diameter[dim] = maxCorner[dim] - minCorner[dim];
            assert(diameter[dim] >= 0_r);
            assert(diameter[dim] > ghostLayerThickness && "ghost layer to larger than subdomain");
        }
    }

    real_t ghostLayerThickness;

    std::array<real_t, 3> minCorner;
    std::array<real_t, 3> maxCorner;

    std::array<real_t, 3> minGhostCorner;
    std::array<real_t, 3> maxGhostCorner;

    std::array<real_t, 3> minInnerCorner;
    std::array<real_t, 3> maxInnerCorner;

    std::array<real_t, 3> diameter;
};

}  // namespace data
}  // namespace mrmd