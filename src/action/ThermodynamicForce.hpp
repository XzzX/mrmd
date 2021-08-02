#pragma once

#include "data/Histogram.hpp"
#include "data/Particles.hpp"
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
    const real_t simulationBoxDiameterX_;
    const real_t simulationBoxDiameterY_;
    const real_t simulationBoxDiameterZ_;
    const real_t thermodynamicForceModulation_;

public:
    inline const auto& getForce() const { return force_; }
    inline const auto& getDensityProfile() const { return densityProfile_; }
    inline const auto& getNumberOfDensityProfileSamples() const { return densityProfileSamples_; }

    void sample(data::Particles& atoms);
    void update();
    void apply(const data::Particles& atoms) const;

    ThermodynamicForce(const real_t targetDensity,
                       const real_t simulationBoxDiameterX,
                       const real_t simulationBoxDiameterY,
                       const real_t simulationBoxDiameterZ,
                       const real_t thermodynamicForceModulation)
        : force_("thermodynamic-force", 0_r, simulationBoxDiameterX, 100),
          densityProfile_("density-profile", 0_r, simulationBoxDiameterX_, 100),
          binVolume_(simulationBoxDiameterY * simulationBoxDiameterZ * densityProfile_.binSize),
          targetDensity_(targetDensity),
          simulationBoxDiameterX_(simulationBoxDiameterX),
          simulationBoxDiameterY_(simulationBoxDiameterY),
          simulationBoxDiameterZ_(simulationBoxDiameterZ),
          thermodynamicForceModulation_(thermodynamicForceModulation)
    {
    }
};
}  // namespace action
}  // namespace mrmd