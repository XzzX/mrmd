#include <gtest/gtest.h>

#include <cassert>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"

namespace mrmd
{
namespace test
{
namespace impl
{
inline void setup(data::Atoms& atoms)
{
    assert(atoms.size() >= 1);
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    auto charge = atoms.getCharge();

    auto policy = Kokkos::RangePolicy<>(0, 1);
    auto kernel = KOKKOS_LAMBDA(idx_t idx)
    {
        pos(0, 0) = real_t(+2);
        pos(0, 1) = real_t(+3);
        pos(0, 2) = real_t(+4);

        vel(0, 0) = real_t(+7);
        vel(0, 1) = real_t(+5);
        vel(0, 2) = real_t(+3);

        force(0, 0) = real_t(+9);
        force(0, 1) = real_t(+7);
        force(0, 2) = real_t(+8);

        mass(0) = real_t(1.5);
        charge(0) = real_t(0.5);
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    atoms.numLocalAtoms = 1;
    atoms.numGhostAtoms = 0;
}
}  // namespace impl
class SingleAtom : public ::testing::Test
{
protected:
    void SetUp() override { impl::setup(atoms); }

    // void TearDown() override {}

    data::Atoms atoms = data::Atoms(1);
};

}  // namespace test
}  // namespace mrmd