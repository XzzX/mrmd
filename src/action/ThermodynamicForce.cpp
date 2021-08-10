#include "ThermodynamicForce.hpp"

#include "analysis/AxialDensityProfile.hpp"
#include "analysis/SmoothenDensityProfile.hpp"

namespace mrmd
{
namespace action
{
void ThermodynamicForce::sample(data::Particles& atoms)
{
    densityProfile_ += analysis::getAxialDensityProfile(atoms.getPos(),
                                                        atoms.numLocalParticles,
                                                        densityProfile_.min,
                                                        densityProfile_.max,
                                                        densityProfile_.numBins);
    ++densityProfileSamples_;
}

void ThermodynamicForce::update()
{
    auto normalizationFactor = 1_r / (binVolume_ * real_c(densityProfileSamples_));

    auto hist = densityProfile_.data;  // avoid capturing this pointer
    auto policy = Kokkos::RangePolicy<>(0, densityProfile_.numBins);
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const idx_t idx) { hist(idx) *= normalizationFactor; },
        "ThermodynamicForce::normalize_sample");
    Kokkos::fence();

    auto smoothedDensityGradient =
        data::gradient(analysis::smoothenDensityProfile(densityProfile_, 2_r, 2_r));
    normalizationFactor = thermodynamicForceModulation_ / targetDensity_;
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const idx_t idx)
        { smoothedDensityGradient.data(idx) *= normalizationFactor; },
        "ThermodynamicForce::calc_force");
    Kokkos::fence();

    force_ += smoothedDensityGradient;

    // reset sampling data
    Kokkos::deep_copy(densityProfile_.data, 0_r);
    densityProfileSamples_ = 0;
}

void ThermodynamicForce::apply(const data::Particles& atoms) const
{
    auto atomsPos = atoms.getPos();
    auto atomsForce = atoms.getForce();

    auto forceHistogram = force_;  // avoid capturing this pointer

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto xPos = atomsPos(idx, 0);
        auto bin = forceHistogram.getBin(xPos);
        if (bin != -1)
        {
            atomsForce(idx, 0) -= forceHistogram.data(bin);
        }
    };
    Kokkos::parallel_for(policy, kernel, "ThermodynamicForce::apply");
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd