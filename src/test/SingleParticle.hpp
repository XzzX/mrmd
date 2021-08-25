#include <gtest/gtest.h>

#include <cassert>

#include "data/Molecules.hpp"
#include "data/Particles.hpp"

namespace mrmd
{
namespace test
{
namespace impl
{
void setup(data::Particles& atoms)
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
        pos(0, 0) = +2_r;
        pos(0, 1) = +3_r;
        pos(0, 2) = +4_r;

        vel(0, 0) = +7_r;
        vel(0, 1) = +5_r;
        vel(0, 2) = +3_r;

        force(0, 0) = +9_r;
        force(0, 1) = +7_r;
        force(0, 2) = +8_r;

        mass(0) = 1.5_r;
        charge(0) = 0.5_r;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    atoms.numLocalParticles = 1;
    atoms.numGhostParticles = 0;
}
}  // namespace impl
class SingleParticle : public ::testing::Test
{
protected:
    void SetUp() override { impl::setup(atoms); }

    // void TearDown() override {}

    data::Particles atoms = data::Particles(1);
};

}  // namespace test
}  // namespace mrmd