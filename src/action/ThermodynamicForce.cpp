#include "ThermodynamicForce.hpp"

#include "analysis/AxialDensityProfile.hpp"
#include "analysis/SmoothenDensityProfile.hpp"

namespace mrmd
{
namespace action
{
void ThermodynamicForce::sample(data::Atoms& atoms)
{
    densityProfile_ += analysis::getAxialDensityProfile(atoms.numLocalAtoms,
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

    auto inputData = densityProfile_.data;  // avoid capturing this pointer
    MultiView hist("hist", densityProfile_.numBins, densityProfile_.numHistograms);
    Kokkos::deep_copy(hist, inputData);
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {densityProfile_.numBins, densityProfile_.numHistograms});
    auto normalizeSampleKernel = KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx)
    {
        inputData(idx, jdx) = 0.5 *
                              (hist(idx, jdx) + hist(densityProfile_.numBins - idx - 1, jdx)) *
                              normalizationFactor;
    };
    Kokkos::parallel_for(policy, normalizeSampleKernel, "ThermodynamicForce::normalize_sample");
    Kokkos::fence();

    auto smoothedDensityGradient = data::gradient(
        analysis::smoothenDensityProfile(densityProfile_, smoothingSigma, smoothingIntensity));
    auto forceFactor = forceFactor_;  ///< avoid capturing this pointer
    auto calcForceKernel = KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
    {
        smoothedDensityGradient.data(binIdx, histogramIdx) *= forceFactor(histogramIdx);
    };
    Kokkos::parallel_for(policy, calcForceKernel, "ThermodynamicForce::calc_force");
    Kokkos::fence();

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

ThermodynamicForce::ThermodynamicForce(const std::vector<real_t>& targetDensity,
                                       const data::Subdomain& subdomain,
                                       const real_t& requestedDensityBinWidth,
                                       const std::vector<real_t>& thermodynamicForceModulation)
    : force_("thermodynamic-force",
             subdomain.minCorner[0],
             subdomain.maxCorner[0],
             idx_c((subdomain.maxCorner[0] - subdomain.minCorner[0]) / requestedDensityBinWidth +
                   0.5_r),
             idx_c(targetDensity.size())),
      densityProfile_(
          "density-profile",
          subdomain.minCorner[0],
          subdomain.maxCorner[0],
          idx_c((subdomain.maxCorner[0] - subdomain.minCorner[0]) / requestedDensityBinWidth +
                0.5_r),
          idx_c(targetDensity.size())),
      binVolume_(subdomain.diameter[1] * subdomain.diameter[2] * densityProfile_.binSize),
      targetDensity_(targetDensity),
      thermodynamicForceModulation_(thermodynamicForceModulation),
      forceFactor_("force-factor", targetDensity.size())
{
    ASSERT_LESS(std::abs(requestedDensityBinWidth - force_.binSize) / requestedDensityBinWidth,
                1e-2,
                "requested bin size is not achieved");

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

}  // namespace action
}  // namespace mrmd