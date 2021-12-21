#pragma once

#include <array>
#include <cassert>

#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
/**
 * |X|O|OOOOOOOOOO
 * |X|O|OOOOOOOOOO
 * |X|O+----------  <-- minInnerCorner
 * |X|OOOOOOOOOOOO
 * |X+------------  <-- minCorner
 * |XXXXXXXXXXXXXX
 * +--------------  <-- minGhostCorner
 */
struct Subdomain
{
    Subdomain() = default;
    Subdomain(const std::array<real_t, 3>& minCornerArg,
              const std::array<real_t, 3>& maxCornerArg,
              real_t ghostLayerThicknessArg)
        : minCorner(minCornerArg),
          maxCorner(maxCornerArg),
          ghostLayerThickness(ghostLayerThicknessArg)
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            minGhostCorner[dim] = minCorner[dim] - ghostLayerThickness;
            maxGhostCorner[dim] = maxCorner[dim] + ghostLayerThickness;

            minInnerCorner[dim] = minCorner[dim] + ghostLayerThickness;
            maxInnerCorner[dim] = maxCorner[dim] - ghostLayerThickness;

            diameter[dim] = maxCorner[dim] - minCorner[dim];
        }
    }

    void scaleDim(const real_t& scalingFactor, const idx_t& dim);
    void scale(const real_t& scalingFactor);

    std::array<real_t, 3> minCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                       std::numeric_limits<real_t>::signaling_NaN(),
                                       std::numeric_limits<real_t>::signaling_NaN()};
    std::array<real_t, 3> maxCorner = {
        std::numeric_limits<real_t>::signaling_NaN(),
        std::numeric_limits<real_t>::signaling_NaN(),
        std::numeric_limits<real_t>::signaling_NaN()};  // namespace data

    real_t ghostLayerThickness = std::numeric_limits<real_t>::signaling_NaN();

    std::array<real_t, 3> minGhostCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN()};
    std::array<real_t, 3> maxGhostCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN()};

    std::array<real_t, 3> minInnerCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN()};
    std::array<real_t, 3> maxInnerCorner = {std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN(),
                                            std::numeric_limits<real_t>::signaling_NaN()};

    std::array<real_t, 3> diameter = {std::numeric_limits<real_t>::signaling_NaN(),
                                      std::numeric_limits<real_t>::signaling_NaN(),
                                      std::numeric_limits<real_t>::signaling_NaN()};
};

void checkInvariants(const Subdomain& subdomain);

}  // namespace data
}  // namespace mrmd