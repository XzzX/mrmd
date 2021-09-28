#include "MeanSquareDisplacement.hpp"

#include "util/Kokkos_grow.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace analysis
{
void MeanSquareDisplacement::reset(data::Atoms& atoms)
{
    numAtoms_ = atoms.numLocalAtoms;
    util::grow(initialPosition_, numAtoms_);

    auto initialPos = initialPosition_;
    auto pos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, numAtoms_);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        initialPos(idx, 0) = pos(idx, 0);
        initialPos(idx, 1) = pos(idx, 1);
        initialPos(idx, 2) = pos(idx, 2);
    };
    Kokkos::parallel_for("MeanSquareDisplacement::reset", policy, kernel);
}

void MeanSquareDisplacement::reset(data::Molecules& molecules)
{
    numAtoms_ = molecules.numLocalMolecules;
    util::grow(initialPosition_, numAtoms_);

    auto initialPos = initialPosition_;
    auto pos = molecules.getPos();

    auto policy = Kokkos::RangePolicy<>(0, numAtoms_);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        initialPos(idx, 0) = pos(idx, 0);
        initialPos(idx, 1) = pos(idx, 1);
        initialPos(idx, 2) = pos(idx, 2);
    };
    Kokkos::parallel_for("MeanSquareDisplacement::reset", policy, kernel);
}

real_t MeanSquareDisplacement::calc(data::Atoms& atoms)
{
    assert(numAtoms_ == atoms.numLocalAtoms);

    auto initialPos = initialPosition_;
    auto pos = atoms.getPos();
    auto subdomain = subdomain_;

    auto sqDisplacement = 0_r;

    auto policy = Kokkos::RangePolicy<>(0, numAtoms_);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, real_t& squareDisplacement)
    {
        real_t dx[3];
        dx[0] = std::abs(initialPos(idx, 0) - pos(idx, 0));
        dx[1] = std::abs(initialPos(idx, 1) - pos(idx, 1));
        dx[2] = std::abs(initialPos(idx, 2) - pos(idx, 2));

        if (dx[0] > 0.5_r * subdomain.diameter[0]) dx[0] -= subdomain.diameter[0];
        if (dx[1] > 0.5_r * subdomain.diameter[1]) dx[1] -= subdomain.diameter[1];
        if (dx[2] > 0.5_r * subdomain.diameter[2]) dx[2] -= subdomain.diameter[2];

        squareDisplacement += util::dot3(dx, dx);
    };
    Kokkos::parallel_reduce("MeanSquareDisplacement::calc", policy, kernel, sqDisplacement);
    return sqDisplacement / real_c(numAtoms_);
}

real_t MeanSquareDisplacement::calc(data::Molecules& molecules)
{
    assert(numAtoms_ == molecules.numLocalMolecules);

    auto initialPos = initialPosition_;
    auto pos = molecules.getPos();
    auto subdomain = subdomain_;

    auto sqDisplacement = 0_r;

    auto policy = Kokkos::RangePolicy<>(0, numAtoms_);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, real_t& squareDisplacement)
    {
        real_t dx[3];
        dx[0] = std::abs(initialPos(idx, 0) - pos(idx, 0));
        dx[1] = std::abs(initialPos(idx, 1) - pos(idx, 1));
        dx[2] = std::abs(initialPos(idx, 2) - pos(idx, 2));

        if (dx[0] > 0.5_r * subdomain.diameter[0]) dx[0] -= subdomain.diameter[0];
        if (dx[1] > 0.5_r * subdomain.diameter[1]) dx[1] -= subdomain.diameter[1];
        if (dx[2] > 0.5_r * subdomain.diameter[2]) dx[2] -= subdomain.diameter[2];

        squareDisplacement += util::dot3(dx, dx);
    };
    Kokkos::parallel_reduce("MeanSquareDisplacement::calc", policy, kernel, sqDisplacement);
    return sqDisplacement / real_c(numAtoms_);
}

}  // namespace analysis
}  // namespace mrmd