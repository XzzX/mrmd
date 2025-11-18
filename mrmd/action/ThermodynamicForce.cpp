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

#include "ThermodynamicForce.hpp"

#include "analysis/AxialDensityProfile.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace action
{
ThermodynamicForce::ThermodynamicForce(const std::vector<real_t>& targetDensity,
                                       const data::Subdomain& subdomain,
                                       const real_t& requestedDensityBinWidth,
                                       const std::vector<real_t>& thermodynamicForceModulation,
                                       const bool enforceSymmetry,
                                       const bool usePeriodicity)
    : force_("thermodynamic-force",
             subdomain.minCorner[0],
             subdomain.maxCorner[0],
             idx_c(std::ceil(subdomain.diameter[0] / requestedDensityBinWidth)),
             idx_c(targetDensity.size())),
      densityProfile_("density-profile", force_),
      binVolume_(subdomain.diameter[1] * subdomain.diameter[2] * densityProfile_.binSize),
      targetDensity_(targetDensity),
      thermodynamicForceModulation_(thermodynamicForceModulation),
      forceFactor_("force-factor", targetDensity.size()),
      enforceSymmetry_(enforceSymmetry),
      usePeriodicity_(usePeriodicity)
{
    MRMD_HOST_CHECK_LESSEQUAL(
        force_.binSize, requestedDensityBinWidth, "requested bin size is not achieved");

    MRMD_HOST_CHECK_EQUAL(targetDensity.size(), thermodynamicForceModulation.size());
    numTypes_ = idx_c(targetDensity.size());
    MRMD_HOST_CHECK_GREATER(numTypes_, 0);

    auto hForceFactor = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forceFactor_);
    for (auto i = 0; i < numTypes_; ++i)
    {
        hForceFactor(i) = thermodynamicForceModulation_[i] / targetDensity_[i];
    }
    Kokkos::deep_copy(forceFactor_, hForceFactor);
}

ThermodynamicForce::ThermodynamicForce(const real_t targetDensity,
                                       const data::Subdomain& subdomain,
                                       const real_t& requestedDensityBinWidth,
                                       const real_t thermodynamicForceModulation,
                                       const bool enforceSymmetry,
                                       const bool usePeriodicity)
    : ThermodynamicForce(std::vector<real_t>{targetDensity},
                         subdomain,
                         requestedDensityBinWidth,
                         {thermodynamicForceModulation},
                         enforceSymmetry,
                         usePeriodicity)
{
}

void ThermodynamicForce::sample(data::Atoms& atoms)
{
    densityProfile_ += analysis::getAxialDensityProfile(atoms.numLocalAtoms,
                                                        atoms.getPos(),
                                                        atoms.getType(),
                                                        numTypes_,
                                                        densityProfile_.min,
                                                        densityProfile_.max,
                                                        densityProfile_.numBins,
                                                        AXIS::X);

    ++densityProfileSamples_;
}

void ThermodynamicForce::update(const real_t& smoothingSigma, const real_t& smoothingIntensity)
{
    MRMD_HOST_CHECK_GREATER(densityProfileSamples_, 0);

    if (enforceSymmetry_)
    {
        densityProfile_.makeSymmetric();
    }

    auto normalizationFactor = 1_r / (binVolume_ * real_c(densityProfileSamples_));
    densityProfile_.scale(normalizationFactor);

    auto smoothedDensityProfile =
        data::smoothen(densityProfile_, smoothingSigma, smoothingIntensity, usePeriodicity_);
    auto smoothedDensityGradient = data::gradient(smoothedDensityProfile, usePeriodicity_);
    smoothedDensityGradient.scale(forceFactor_);

    force_ -= smoothedDensityGradient;

    // reset sampling data
    Kokkos::deep_copy(densityProfile_.data, 0_r);
    densityProfileSamples_ = 0;
}

void ThermodynamicForce::apply(const data::Atoms& atoms) const
{
    apply_if(atoms, KOKKOS_LAMBDA(const real_t, const real_t, const real_t) { return true; });
}

void ThermodynamicForce::apply(const data::Atoms& atoms, const weighting_function::Slab& slab) const
{
    apply_if(
        atoms, KOKKOS_LAMBDA(const real_t x, const real_t y, const real_t z) {
            return slab.isInHYRegion(x, y, z);
        });
}

std::vector<real_t> ThermodynamicForce::getMuLeft() const
{
    auto Fth = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), getForce().data);

    std::vector<real_t> muLeft(numTypes_, 0_r);
    for (auto typeId = 0; typeId < numTypes_; ++typeId)
    {
        for (size_t i = 0; i < Fth.extent(0) / 2; ++i)
        {
            muLeft[typeId] += Fth(i, typeId);
        }
        muLeft[typeId] *= getForce().binSize;
    }

    return muLeft;
}

std::vector<real_t> ThermodynamicForce::getMuRight() const
{
    auto Fth = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), getForce().data);

    std::vector<real_t> muLeft(numTypes_, 0_r);
    for (auto typeId = 0; typeId < numTypes_; ++typeId)
    {
        for (size_t i = Fth.extent(0) / 2; i < Fth.extent(0); ++i)
        {
            muLeft[typeId] += Fth(i, typeId);
        }
        muLeft[typeId] *= getForce().binSize;
    }

    return muLeft;
}

}  // namespace action
}  // namespace mrmd