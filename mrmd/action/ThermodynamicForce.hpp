// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "assert/assert.hpp"
#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "util/ApplicationRegion.hpp"
#include "util/interpolation.hpp"
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
    const std::vector<real_t> targetDensities_;
    const real_t densityBinWidth_;
    const std::vector<real_t> thermodynamicForceModulations_;
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
    inline void setForce(const MultiView& forces) const { Kokkos::deep_copy(force_.data, forces); }
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
    void update(const real_t& smoothingSigma, const real_t& smoothingIntensity, const util::ApplicationRegion& applicationRegion);
    void apply(const data::Atoms& atoms) const;
    void apply(const data::Atoms& atoms, const weighting_function::Slab& slab) const;
    void apply(const data::Atoms& atoms, const util::ApplicationRegion& applicationRegion) const;
    std::vector<real_t> getMuLeft() const;
    std::vector<real_t> getMuRight() const;

    ThermodynamicForce(const std::vector<real_t>& targetDensities,
                       const data::Subdomain& subdomain,
                       const real_t& requestedDensityGridSpacing,
                       const real_t& requestedDensityBinWidth,
                       const std::vector<real_t>& thermodynamicForceModulations,
                       const bool enforceSymmetry = false,
                       const bool usePeriodicity = false);

    ThermodynamicForce(const std::vector<real_t>& targetDensities,
                       const data::Subdomain& subdomain,
                       const real_t& requestedDensityGridSpacing,
                       const std::vector<real_t>& thermodynamicForceModulations,
                       const bool enforceSymmetry = false,
                       const bool usePeriodicity = false);

    ThermodynamicForce(const real_t& targetDensity,
                       const data::Subdomain& subdomain,
                       const real_t& requestedDensityGridSpacing,
                       const real_t& thermodynamicForceModulation,
                       const bool enforceSymmetry = false,
                       const bool usePeriodicity = false);

    ThermodynamicForce(const std::vector<real_t>& targetDensities,
                       const data::Subdomain& subdomain,
                       const idx_t& requestedDensityBinNumber,
                       const real_t& requestedDensityBinWidth,
                       const std::vector<real_t>& thermodynamicForceModulations,
                       const bool enforceSymmetry = false,
                       const bool usePeriodicity = false);
};
}  // namespace action
}  // namespace mrmd