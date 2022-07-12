#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"

namespace mrmd
{
namespace test
{
namespace impl
{
inline void setupMolecules(data::Molecules& molecules)
{
    assert(molecules.size() >= 2);

    auto moleculesAtomsOffset = molecules.getAtomsOffset();
    auto moleculesNumAtoms = molecules.getNumAtoms();

    auto policy = Kokkos::RangePolicy<>(0, 2);
    auto kernel = KOKKOS_LAMBDA(idx_t idx)
    {
        if (idx == 0)
        {
            moleculesAtomsOffset(0) = 0;
            moleculesNumAtoms(0) = 2;
        }
        if (idx == 1)
        {
            moleculesAtomsOffset(1) = 2;
            moleculesNumAtoms(1) = 2;
        }
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    molecules.numLocalMolecules = 2;
    molecules.numGhostMolecules = 0;
}

inline void setupAtoms(data::Atoms& atoms)
{
    assert(atoms.size() >= 4);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    auto relativeMass = atoms.getRelativeMass();

    auto policy = Kokkos::RangePolicy<>(0, 4);
    auto kernel = KOKKOS_LAMBDA(idx_t idx)
    {
        if (idx == 0)
        {
            pos(0, 0) = real_t(1);
            pos(0, 1) = real_t(0);
            pos(0, 2) = real_t(0);
            mass(0) = real_t(1);
            relativeMass(0) = real_t(0.25);
        }

        if (idx == 1)
        {
            pos(1, 0) = real_t(0);
            pos(1, 1) = real_t(1);
            pos(1, 2) = real_t(0);
            mass(1) = real_t(3);
            relativeMass(1) = real_t(0.75);
        }
        if (idx == 2)
        {
            pos(2, 0) = real_t(-1);
            pos(2, 1) = real_t(0);
            pos(2, 2) = real_t(0);
            mass(2) = real_t(1);
            relativeMass(2) = real_t(0.25);
        }

        if (idx == 3)
        {
            pos(3, 0) = real_t(0);
            pos(3, 1) = real_t(-1);
            pos(3, 2) = real_t(0);
            mass(3) = real_t(3);
            relativeMass(3) = real_t(0.75);
        }
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    atoms.numLocalAtoms = 4;
    atoms.numGhostAtoms = 0;
}
}  // namespace impl

/**
 * 2 molecules with 2 atoms each
 * +++++
 * ++A++
 * +B+A+
 * ++B++
 * +++++
 */
class DiamondFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        impl::setupMolecules(molecules);
        impl::setupAtoms(atoms);
    }

    // void TearDown() override {}

    data::Molecules molecules = data::Molecules(2);
    data::Atoms atoms = data::Atoms(4);
};

}  // namespace test
}  // namespace mrmd