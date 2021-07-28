#pragma once

#include <Kokkos_Random.hpp>

#include "data/Histogram.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class ThermodynamicForce
{
public:
    static void apply(const data::Particles& atoms, const data::Histogram& force)
    {
        auto atomsPos = atoms.getPos();
        auto atomsForce = atoms.getForce();

        auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalParticles);
        auto kernel = KOKKOS_LAMBDA(const idx_t idx)
        {
            auto xPos = atomsPos(idx, 0);
            auto bin = idx_c((xPos - force.min) * force.inverseBinSize);
            bin = std::max(idx_t(0), bin);
            bin = std::min(force.numBins - 1, bin);
            atomsForce(idx, 0) -= force.data(bin);
        };
        Kokkos::parallel_for(policy, kernel, "ThermodynamicForce::apply");
        Kokkos::fence();
    }
};
}  // namespace action
}  // namespace mrmd