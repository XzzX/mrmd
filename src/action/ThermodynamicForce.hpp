#pragma once

#include "assert/assert.hpp"
#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "weighting_function/Slab.hpp"

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
    bool enforceSymmetry_ = false;
    bool usePeriodicity_ = false;

public:
    inline auto getForce() const { return force_; }
    inline auto getForce(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        return Kokkos::subview(force_.data, Kokkos::ALL(), typeId);
    }

    inline auto getDensityProfile() const { return densityProfile_; }
    inline auto getDensityProfile(const idx_t& typeId) const
    {
        assert(typeId < numTypes_);
        assert(typeId >= 0);
        return Kokkos::subview(densityProfile_.data, Kokkos::ALL(), typeId);
    }
    inline const auto& getNumberOfDensityProfileSamples() const { return densityProfileSamples_; }

    void sample(data::Atoms& atoms);
    void update(const real_t& smoothingSigma, const real_t& smoothingIntensity);
    void apply(const data::Atoms& atoms) const;
    void apply(const data::Atoms& atoms, const weighting_function::Slab& slab) const;

    ThermodynamicForce(const std::vector<real_t>& targetDensity,
                       const data::Subdomain& subdomain,
                       const real_t& requestedDensityBinWidth,
                       const std::vector<real_t>& thermodynamicForceModulation,
                       const bool enforceSymmetry = false,
                       const bool usePeriodicity = false);

    ThermodynamicForce(const real_t targetDensity,
                       const data::Subdomain& subdomain,
                       const real_t& requestedDensityBinWidth,
                       const real_t thermodynamicForceModulation,
                       const bool enforceSymmetry = false,
                       const bool usePeriodicity = false);
};
}  // namespace action
}  // namespace mrmd