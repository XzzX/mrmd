#pragma once

#include "data/Atoms.hpp"
#include "data/Histogram.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class ThermodynamicForce
{
private:
    data::Histogram force_;
    data::Histogram densityProfile_;
    idx_t densityProfileSamples_ = 0;
    const real_t binVolume_;
    const real_t targetDensity_;
    const real_t thermodynamicForceModulation_;

public:
    inline const auto& getForce() const { return force_; }
    inline const auto& getDensityProfile() const { return densityProfile_; }
    inline const auto& getNumberOfDensityProfileSamples() const { return densityProfileSamples_; }

    void sample(data::Atoms& atoms);
    void update();
    void apply(const data::Atoms& atoms) const;

    ThermodynamicForce(const real_t targetDensity,
                       const data::Subdomain& subdomain,
                       const real_t thermodynamicForceModulation)
        : force_("thermodynamic-force", subdomain.minCorner[0], subdomain.maxCorner[0], 100),
          densityProfile_("density-profile", subdomain.minCorner[0], subdomain.maxCorner[0], 100),
          binVolume_(subdomain.diameter[1] * subdomain.diameter[2] * densityProfile_.binSize),
          targetDensity_(targetDensity),
          thermodynamicForceModulation_(thermodynamicForceModulation)
    {
    }
};
}  // namespace action
}  // namespace mrmd