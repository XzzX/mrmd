#include "MeanSquareDisplacement.hpp"

#include "util/Kokkos_grow.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace analysis
{
void MeanSquareDisplacement::reset(data::Particles& atoms)
{
    numParticles_ = atoms.numLocalParticles;
    util::grow(initialPosition_, numParticles_);

    auto initialPos = initialPosition_;
    auto pos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, numParticles_);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        initialPos(idx, 0) = pos(idx, 0);
        initialPos(idx, 1) = pos(idx, 1);
        initialPos(idx, 2) = pos(idx, 2);
    };
    Kokkos::parallel_for("MeanSquareDisplacement::reset", policy, kernel);
}
real_t MeanSquareDisplacement::calc(data::Particles& atoms)
{
    auto initialPos = initialPosition_;
    auto pos = atoms.getPos();

    auto sqDisplacement = 0_r;

    auto policy = Kokkos::RangePolicy<>(0, numParticles_);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, real_t& squareDisplacement)
    {
        real_t dx[3];
        dx[0] = initialPos(idx, 0) - pos(idx, 0);
        dx[1] = initialPos(idx, 1) - pos(idx, 1);
        dx[2] = initialPos(idx, 2) - pos(idx, 2);
        squareDisplacement += util::dot3(dx, dx);
    };
    Kokkos::parallel_reduce("MeanSquareDisplacement::calc", policy, kernel, sqDisplacement);
    return sqDisplacement / real_c(numParticles_);
}

}  // namespace analysis
}  // namespace mrmd