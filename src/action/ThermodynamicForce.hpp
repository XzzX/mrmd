#pragma once

#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class ThermodynamicForce
{
private:
    data::MultiHistogram force_;
    data::MultiHistogram densityProfile_;
    idx_t densityProfileSamples_ = 0;
    real_t binVolume_;
    const std::vector<real_t> targetDensity_;
    const std::vector<real_t> thermodynamicForceModulation_;
    idx_t numTypes_;

    ScalarView forceFactor_;  ///< precalculated prefactor for force calculation

public:
    inline auto getForce() const { return force_; }
    inline auto getForce(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        return Kokkos::subview(force_.data, Kokkos::ALL(), typeId);
    }
    inline auto getDensityProfile(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        return Kokkos::subview(densityProfile_.data, Kokkos::ALL(), typeId);
    }
    inline const auto& getNumberOfDensityProfileSamples() const { return densityProfileSamples_; }

    void sample(data::Atoms& atoms);
    void update();
    void apply(const data::Atoms& atoms) const;

    ThermodynamicForce(const std::vector<real_t>& targetDensity,
                       const data::Subdomain& subdomain,
                       const std::vector<real_t>& thermodynamicForceModulation)
        : force_("thermodynamic-force",
                 subdomain.minCorner[0],
                 subdomain.maxCorner[0],
                 100,
                 idx_c(targetDensity.size())),
          densityProfile_("density-profile",
                          subdomain.minCorner[0],
                          subdomain.maxCorner[0],
                          100,
                          idx_c(targetDensity.size())),
          binVolume_(subdomain.diameter[1] * subdomain.diameter[2] * densityProfile_.binSize),
          targetDensity_(targetDensity),
          thermodynamicForceModulation_(thermodynamicForceModulation),
          forceFactor_("force-factor", targetDensity.size())
    {
        assert(targetDensity.size() == thermodynamicForceModulation.size());
        numTypes_ = idx_c(targetDensity.size());
        assert(numTypes_ > 0);

        auto hForceFactor = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forceFactor_);
        for (auto i = 0; i < numTypes_; ++i)
        {
            hForceFactor(i) = thermodynamicForceModulation_[i] / targetDensity_[i];
        }
        Kokkos::deep_copy(forceFactor_, hForceFactor);
    }

    ThermodynamicForce(const real_t targetDensity,
                       const data::Subdomain& subdomain,
                       const real_t thermodynamicForceModulation)
        : ThermodynamicForce(
              std::vector<real_t>{targetDensity}, subdomain, {thermodynamicForceModulation})
    {
    }
};
}  // namespace action
}  // namespace mrmd