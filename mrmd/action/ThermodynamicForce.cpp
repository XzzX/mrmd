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

ThermodynamicForce::ThermodynamicForce(const std::vector<real_t>& targetDensities,
                                       const data::Subdomain& subdomain,
                                       const real_t& requestedDensityGridSpacing,
                                       const real_t& requestedDensityBinWidth,
                                       const std::vector<real_t>& thermodynamicForceModulations,
                                       const bool enforceSymmetry,
                                       const bool usePeriodicity)
    : force_("thermodynamic-force",
             subdomain.minCorner[0],
             subdomain.maxCorner[0],
             idx_c(std::ceil(subdomain.diameter[0] / requestedDensityGridSpacing)),
             idx_c(targetDensities.size())),
      densityProfile_("density-profile",
                      subdomain.minCorner[0],
                      subdomain.maxCorner[0],
                      idx_c(std::ceil(subdomain.diameter[0] / requestedDensityGridSpacing)),
                      idx_c(targetDensities.size())),
      binVolume_(subdomain.diameter[1] * subdomain.diameter[2] * densityProfile_.binSize),
      targetDensities_(targetDensities),
      densityBinWidth_(requestedDensityBinWidth),
      thermodynamicForceModulations_(thermodynamicForceModulations),
      forceFactor_("force-factor", targetDensities.size()),
      enforceSymmetry_(enforceSymmetry),
      usePeriodicity_(usePeriodicity)
{
    MRMD_HOST_CHECK_LESSEQUAL(force_.binSize,
                              requestedDensityGridSpacing,
                              "requested thermodynamic force bin size is not achieved");
    MRMD_HOST_CHECK_LESSEQUAL(densityProfile_.binSize,
                              requestedDensityGridSpacing,
                              "requested density profile bin size is not achieved");

    MRMD_HOST_CHECK_EQUAL(targetDensities.size(), thermodynamicForceModulations.size());
    numTypes_ = idx_c(targetDensities.size());
    MRMD_HOST_CHECK_GREATER(numTypes_, 0);

    auto hForceFactor = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forceFactor_);
    for (auto i = 0; i < numTypes_; ++i)
    {
        hForceFactor(i) = thermodynamicForceModulations_[i] / targetDensities_[i];
    }
    Kokkos::deep_copy(forceFactor_, hForceFactor);

    std::cout << "densGridSpacing: " << requestedDensityGridSpacing << std::endl;
    std::cout << "densBinWidth: " << requestedDensityBinWidth << std::endl;
}

ThermodynamicForce::ThermodynamicForce(const std::vector<real_t>& targetDensities,
                                       const data::Subdomain& subdomain,
                                       const real_t& requestedDensityGridSpacing,
                                       const std::vector<real_t>& thermodynamicForceModulations,
                                       const bool enforceSymmetry,
                                       const bool usePeriodicity)
    : ThermodynamicForce(targetDensities,
                         subdomain,
                         requestedDensityGridSpacing,
                         requestedDensityGridSpacing,
                         thermodynamicForceModulations,
                         enforceSymmetry,
                         usePeriodicity)
{
}

ThermodynamicForce::ThermodynamicForce(const real_t& targetDensity,
                                       const data::Subdomain& subdomain,
                                       const real_t& requestedDensityGridSpacing,
                                       const real_t& thermodynamicForceModulation,
                                       const bool enforceSymmetry,
                                       const bool usePeriodicity)
    : ThermodynamicForce({targetDensity},
                         subdomain,
                         requestedDensityGridSpacing,
                         requestedDensityGridSpacing,
                         {thermodynamicForceModulation},
                         enforceSymmetry,
                         usePeriodicity)
{
}

ThermodynamicForce::ThermodynamicForce(const std::vector<real_t>& targetDensities,
                                       const data::Subdomain& subdomain,
                                       const idx_t& requestedDensityBinNumber,
                                       const real_t& requestedDensityBinWidth,
                                       const std::vector<real_t>& thermodynamicForceModulations,
                                       const bool enforceSymmetry,
                                       const bool usePeriodicity)
    : force_("thermodynamic-force",
             subdomain.minCorner[0],
             subdomain.maxCorner[0],
             requestedDensityBinNumber,
             idx_c(targetDensities.size())),
      densityProfile_("density-profile",
                      subdomain.minCorner[0],
                      subdomain.maxCorner[0],
                      requestedDensityBinNumber,
                      idx_c(targetDensities.size())),
      binVolume_(subdomain.diameter[1] * subdomain.diameter[2] * densityProfile_.binSize),
      targetDensities_(targetDensities),
      densityBinWidth_(requestedDensityBinWidth),
      thermodynamicForceModulations_(thermodynamicForceModulations),
      forceFactor_("force-factor", targetDensities.size()),
      enforceSymmetry_(enforceSymmetry),
      usePeriodicity_(usePeriodicity)
{
    MRMD_HOST_CHECK_EQUAL(targetDensities.size(), thermodynamicForceModulations.size());
    numTypes_ = idx_c(targetDensities.size());
    MRMD_HOST_CHECK_GREATER(numTypes_, 0);

    auto hForceFactor = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), forceFactor_);
    for (auto i = 0; i < numTypes_; ++i)
    {
        hForceFactor(i) = thermodynamicForceModulations_[i] / targetDensities_[i];
    }
    Kokkos::deep_copy(forceFactor_, hForceFactor);
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
                                                        densityBinWidth_,
                                                        COORD_X);

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

void ThermodynamicForce::update(const real_t& smoothingSigma, const real_t& smoothingIntensity, const util::ApplicationRegion& applicationRegion)
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
    
    auto constrainedDensityGradient = util::constrainToApplicationRegion(smoothedDensityGradient, applicationRegion);

    force_ -= util::interpolate(constrainedDensityGradient, force_.createGrid_d());

    // reset sampling data
    Kokkos::deep_copy(densityProfile_.data, 0_r);
    densityProfileSamples_ = 0;
}

void ThermodynamicForce::apply(const data::Atoms& atoms) const
{
    auto atomsPos = atoms.getPos();
    auto atomsForce = atoms.getForce();
    auto atomsType = atoms.getType();

    auto forceHistogram = force_;  // avoid capturing this pointer

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto xPos = atomsPos(idx, 0);
        auto bin = forceHistogram.getBin(xPos);
        if (bin != -1)
        {
            MRMD_DEVICE_ASSERT_LESS(atomsType(idx), forceHistogram.numHistograms);
            MRMD_DEVICE_ASSERT(!std::isnan(forceHistogram.data(bin, atomsType(idx))));
            atomsForce(idx, 0) += forceHistogram.data(bin, atomsType(idx));
        }
    };
    Kokkos::parallel_for("ThermodynamicForce::apply", policy, kernel);
    Kokkos::fence();
}

void ThermodynamicForce::apply(const data::Atoms& atoms, const weighting_function::Slab& slab) const
{
    auto atomsPos = atoms.getPos();
    auto atomsForce = atoms.getForce();
    auto atomsType = atoms.getType();

    auto forceHistogram = force_;  // avoid capturing this pointer

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto xPos = atomsPos(idx, 0);
        if (!slab.isInHYRegion(atomsPos(idx, 0), atomsPos(idx, 1), atomsPos(idx, 2))) return;
        auto bin = forceHistogram.getBin(xPos);
        if (bin != -1)
        {
            MRMD_DEVICE_ASSERT_LESS(atomsType(idx), forceHistogram.numHistograms);
            MRMD_DEVICE_ASSERT(!std::isnan(forceHistogram.data(bin, atomsType(idx))));
            atomsForce(idx, 0) += forceHistogram.data(bin, atomsType(idx));
        }
    };
    Kokkos::parallel_for("ThermodynamicForce::apply", policy, kernel);
    Kokkos::fence();
}

void ThermodynamicForce::apply(const data::Atoms& atoms,
                               const util::ApplicationRegion& applicationRegion) const
{
    auto atomsPos = atoms.getPos();
    auto atomsForce = atoms.getForce();
    auto atomsType = atoms.getType();

    auto forceHistogram = force_;  // avoid capturing this pointer

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto xPos = atomsPos(idx, 0);
        if (!applicationRegion.isInApplicationRegion(
                atomsPos(idx, 0), atomsPos(idx, 1), atomsPos(idx, 2)))
            return;
        auto bin = forceHistogram.getBin(xPos);
        if (bin != -1)
        {
            MRMD_DEVICE_ASSERT_LESS(atomsType(idx), forceHistogram.numHistograms);
            MRMD_DEVICE_ASSERT(!std::isnan(forceHistogram.data(bin, atomsType(idx))));
            atomsForce(idx, 0) += forceHistogram.data(bin, atomsType(idx));
        }
    };
    Kokkos::parallel_for("ThermodynamicForce::apply", policy, kernel);
    Kokkos::fence();
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