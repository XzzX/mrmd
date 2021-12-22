#include "ThermodynamicForce.hpp"

#include "analysis/AxialDensityProfile.hpp"
#include "analysis/SmoothenDensityProfile.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace action
{
ThermodynamicForce::ThermodynamicForce(const std::vector<real_t>& targetDensity,
                                       const data::Subdomain& subdomain,
                                       const real_t& requestedDensityBinWidth,
                                       const std::vector<real_t>& thermodynamicForceModulation)
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
      forceFactor_("force-factor", targetDensity.size())
{
    ASSERT_LESS(force_.binSize, requestedDensityBinWidth, "requested bin size is not achieved");

    ASSERT_EQUAL(targetDensity.size(), thermodynamicForceModulation.size());
    numTypes_ = idx_c(targetDensity.size());
    ASSERT_GREATER(numTypes_, 0);

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
                                       const real_t thermodynamicForceModulation)
    : ThermodynamicForce(std::vector<real_t>{targetDensity},
                         subdomain,
                         requestedDensityBinWidth,
                         {thermodynamicForceModulation})
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
    assert(densityProfileSamples_ > 0);

    auto normalizationFactor = 1_r / (binVolume_ * real_c(densityProfileSamples_));
    densityProfile_.scale(normalizationFactor);

    auto smoothedDensityGradient = data::gradient(
        analysis::smoothenDensityProfile(densityProfile_, smoothingSigma, smoothingIntensity));
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
            assert(atomsType(idx) < forceHistogram.numHistograms);
            assert(!std::isnan(forceHistogram.data(bin, atomsType(idx))));
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
            assert(atomsType(idx) < forceHistogram.numHistograms);
            assert(!std::isnan(forceHistogram.data(bin, atomsType(idx))));
            atomsForce(idx, 0) -= forceHistogram.data(bin, atomsType(idx));
        }
    };
    Kokkos::parallel_for(policy, kernel, "ThermodynamicForce::apply");
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd