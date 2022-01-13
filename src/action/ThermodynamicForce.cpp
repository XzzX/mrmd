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
                                       const bool enforceSymmetry)
    : force_("thermodynamic-force",
             subdomain.minGhostCorner[0],
             subdomain.maxGhostCorner[0],
             idx_c(std::ceil(subdomain.diameterWithGhostLayer[0] / requestedDensityBinWidth)),
             idx_c(targetDensity.size())),
      densityProfile_("density-profile", force_),
      binVolume_(subdomain.diameterWithGhostLayer[1] * subdomain.diameterWithGhostLayer[2] *
                 densityProfile_.binSize),
      targetDensity_(targetDensity),
      thermodynamicForceModulation_(thermodynamicForceModulation),
      forceFactor_("force-factor", targetDensity.size()),
      enforceSymmetry_(enforceSymmetry)
{
    MRMD_HOST_CHECK_LESS(
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
                                       const bool enforceSymmetry)
    : ThermodynamicForce(std::vector<real_t>{targetDensity},
                         subdomain,
                         requestedDensityBinWidth,
                         {thermodynamicForceModulation},
                         enforceSymmetry)
{
}

void ThermodynamicForce::sample(data::Atoms& atoms)
{
    densityProfile_ += analysis::getAxialDensityProfile(atoms.numLocalAtoms + atoms.numGhostAtoms,
                                                        atoms.getPos(),
                                                        atoms.getType(),
                                                        numTypes_,
                                                        densityProfile_.min,
                                                        densityProfile_.max,
                                                        densityProfile_.numBins);

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

    auto smoothedDensityGradient =
        data::gradient(data::smoothen(densityProfile_, smoothingSigma, smoothingIntensity));
    smoothedDensityGradient.scale(forceFactor_);

    force_ += smoothedDensityGradient;

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
            atomsForce(idx, 0) -= forceHistogram.data(bin, atomsType(idx));
        }
    };
    Kokkos::parallel_for(policy, kernel, "ThermodynamicForce::apply");
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
            atomsForce(idx, 0) -= forceHistogram.data(bin, atomsType(idx));
        }
    };
    Kokkos::parallel_for(policy, kernel, "ThermodynamicForce::apply");
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd