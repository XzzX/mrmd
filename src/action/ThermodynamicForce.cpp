#include "ThermodynamicForce.hpp"

#include "analysis/AxialDensityProfile.hpp"
#include "analysis/SmoothenDensityProfile.hpp"

namespace mrmd
{
namespace action
{
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

}  // namespace action
}  // namespace mrmd