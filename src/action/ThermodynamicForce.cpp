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
    auto normalizationFactor = 1_r / (binVolume_ * real_c(densityProfileSamples_)) *
                               thermodynamicForceModulation_ / targetDensity_;
    auto policy = Kokkos::RangePolicy<>(0, densityProfile_.numBins);
    Kokkos::parallel_for(
        policy,
        KOKKOS_LAMBDA(const idx_t idx) { densityProfile_.data(idx) *= normalizationFactor; },
        "ThermodynamicForce::update");
    Kokkos::fence();
    auto smoothedDensityProfile = analysis::smoothenDensityProfile(densityProfile_, 3_r, 6_r);

    force_ += data::gradient(smoothedDensityProfile);

    // reset sampling data
    Kokkos::deep_copy(densityProfile_.data, 0_r);
    densityProfileSamples_ = 0;
}

void ThermodynamicForce::apply(const data::Particles& atoms) const
{
    auto atomsPos = atoms.getPos();
    auto atomsForce = atoms.getForce();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto xPos = atomsPos(idx, 0);
        auto bin = idx_c((xPos - force_.min) * force_.inverseBinSize);
        bin = std::max(idx_t(0), bin);
        bin = std::min(force_.numBins - 1, bin);
        atomsForce(idx, 0) -= force_.data(bin);
    };
    Kokkos::parallel_for(policy, kernel, "ThermodynamicForce::apply");
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd