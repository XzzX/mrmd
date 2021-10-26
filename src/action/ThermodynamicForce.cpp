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

void ThermodynamicForce::update()
{
    auto normalizationFactor = 1_r / (binVolume_ * real_c(densityProfileSamples_));

    auto hist = densityProfile_.data;  // avoid capturing this pointer
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {idx_t(0), idx_t(0)}, {densityProfile_.numBins, densityProfile_.numHistograms});
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const idx_t idx, const idx_t jdx) { hist(idx, jdx) *= normalizationFactor; },
        "ThermodynamicForce::normalize_sample");
    Kokkos::fence();

    auto smoothedDensityGradient =
        data::gradient(analysis::smoothenDensityProfile(densityProfile_, 2_r, 2_r));
    normalizationFactor = thermodynamicForceModulation_[0] / targetDensity_[0];
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const idx_t binIdx, const idx_t histogramIdx)
        { smoothedDensityGradient.data(binIdx, histogramIdx) *= normalizationFactor; },
        "ThermodynamicForce::calc_force");
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
            atomsForce(idx, 0) -= forceHistogram.data(bin, atomsType(idx));
        }
    };
    Kokkos::parallel_for(policy, kernel, "ThermodynamicForce::apply");
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd