#include "KineticEnergy.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"

namespace mrmd
{
TEST(KineticEnergy, Simple)
{
    data::Atoms atoms(3);
    auto d_AoSoA = atoms.getAoSoA();
    auto h_AoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), d_AoSoA);

    auto vel = Cabana::slice<data::Atoms::VEL>(h_AoSoA);
    auto mass = Cabana::slice<data::Atoms::MASS>(h_AoSoA);

    vel(0, 0) = real_t(+2);
    vel(0, 1) = real_t(+0);
    vel(0, 2) = real_t(+0);
    mass(0) = real_t(1);
    vel(1, 0) = real_t(-0);
    vel(1, 1) = real_t(-8);
    vel(1, 2) = real_t(-0);
    mass(1) = real_t(2);
    vel(2, 0) = real_t(+0);
    vel(2, 1) = real_t(+0);
    vel(2, 2) = real_t(+16);
    mass(2) = real_t(0.5);

    Cabana::deep_copy(d_AoSoA, h_AoSoA);

    atoms.numLocalAtoms = 3;
    atoms.numGhostAtoms = 0;

    auto energy = analysis::getKineticEnergy(atoms);

    EXPECT_FLOAT_EQ(energy,
                    (real_t(4) + real_t(2) * real_t(64) + real_t(0.5) * real_t(256)) * real_t(0.5));
}
}  // namespace mrmd