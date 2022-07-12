#include "LJ_IdealGas.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"

namespace mrmd
{
namespace action
{
/**
 * 2 molecules with 2 atoms each
 * ++++++
 * +A++B+
 * ++++++
 * +A++B+
 * ++++++
 */
class LJ_IdealGas_Test : public ::testing::Test
{
protected:
    static auto getMolecules()
    {
        data::HostMolecules molecules(2);

        auto hMolecules =
            Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), molecules.getAoSoA());

        auto pos = Cabana::slice<data::Molecules::POS>(hMolecules);
        pos(0, 0) = real_t(-0.5);
        pos(0, 1) = real_t(0);
        pos(0, 2) = real_t(0);

        pos(1, 0) = real_t(+0.5);
        pos(1, 1) = real_t(0);
        pos(1, 2) = real_t(0);

        auto moleculesAtomsOffset = Cabana::slice<data::Molecules::ATOMS_OFFSET>(hMolecules);
        auto moleculesNumAtoms = Cabana::slice<data::Molecules::NUM_ATOMS>(hMolecules);
        moleculesAtomsOffset(0) = 0;
        moleculesNumAtoms(0) = 2;
        moleculesAtomsOffset(1) = 2;
        moleculesNumAtoms(1) = 2;
        Cabana::deep_copy(molecules.getAoSoA(), hMolecules);

        molecules.numLocalMolecules = 2;

        return molecules;
    }

    static auto getAtoms()
    {
        data::HostAtoms atoms(4);

        auto hAtoms = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
        auto pos = Cabana::slice<data::Atoms::POS>(hAtoms);
        auto relativeMass = Cabana::slice<data::Atoms::RELATIVE_MASS>(hAtoms);

        pos(0, 0) = real_t(-0.5);
        pos(0, 1) = real_t(-0.5);
        pos(0, 2) = real_t(0);
        relativeMass(0) = real_t(0.5);

        pos(1, 0) = real_t(-0.5);
        pos(1, 1) = real_t(+0.5);
        pos(1, 2) = real_t(0);
        relativeMass(1) = real_t(0.5);

        pos(2, 0) = real_t(+0.5);
        pos(2, 1) = real_t(-0.5);
        pos(2, 2) = real_t(0);
        relativeMass(2) = real_t(0.5);

        pos(3, 0) = real_t(+0.5);
        pos(3, 1) = real_t(+0.5);
        pos(3, 2) = real_t(0);
        relativeMass(3) = real_t(0.5);

        Cabana::deep_copy(atoms.getAoSoA(), hAtoms);

        auto type = atoms.getType();
        Cabana::deep_copy(type, 0);

        atoms.numLocalAtoms = 4;

        return atoms;
    }

    void SetUp() override
    {
        data::deep_copy(molecules, getMolecules());

        auto cutoff = real_t(2);
        auto cellRatio = real_t(1);
        real_t minGrid[3] = {real_t(-1), real_t(-1), real_t(-1)};
        real_t maxGrid[3] = {real_t(+1), real_t(+1), real_t(+1)};
        auto expectedNumNeighbors = 4;
        moleculesVerletList.build(molecules.getPos(),
                                  0,
                                  molecules.numLocalMolecules,
                                  cutoff,
                                  cellRatio,
                                  minGrid,
                                  maxGrid,
                                  expectedNumNeighbors);

        data::deep_copy(atoms, getAtoms());
        auto atomsForce = atoms.getForce();
        Cabana::deep_copy(atomsForce, real_t(0));
    }

    // void TearDown() override {}

    static constexpr real_t epsilon = real_t(2);
    static constexpr real_t sigma = real_t(0.9);
    static constexpr real_t rc = real_t(2.5) * sigma;
    static constexpr real_t cappingDistance = real_t(0);
    static constexpr real_t eps = real_t(0.001);

    data::Molecules molecules = data::Molecules(1);
    HalfVerletList moleculesVerletList;
    data::Atoms atoms = data::Atoms(1);
};

TEST_F(LJ_IdealGas_Test, CG)
{
    auto atomsForce = atoms.getForce();
    Cabana::deep_copy(atomsForce, real_t(2));

    auto moleculesLambda = molecules.getModulatedLambda();
    Cabana::deep_copy(moleculesLambda, real_t(0));
    action::LJ_IdealGas LJ(cappingDistance, rc, sigma, epsilon, true);
    LJ.run(molecules, moleculesVerletList, atoms);

    data::HostAtoms h_atoms(atoms);
    for (idx_t idx = 0; idx < 4; ++idx)
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            EXPECT_FLOAT_EQ(h_atoms.getForce()(idx, dim), real_t(2));
        }
    }
}

TEST_F(LJ_IdealGas_Test, HY)
{
    auto moleculesLambda = molecules.getModulatedLambda();
    Cabana::deep_copy(moleculesLambda, real_t(0.5));

    action::LJ_IdealGas LJ(cappingDistance, rc, sigma, epsilon, true);
    LJ.run(molecules, moleculesVerletList, atoms);

    constexpr auto xForce = real_t(0.22156665) * real_t(0.5);
    constexpr auto yForce = real_t(1.3825009) * real_t(0.5);

    data::HostAtoms h_atoms(atoms);
    auto force = h_atoms.getForce();

    EXPECT_FLOAT_EQ(force(0, 0), -xForce);
    EXPECT_FLOAT_EQ(force(0, 1), +yForce);
    EXPECT_FLOAT_EQ(force(0, 2), real_t(0));

    EXPECT_FLOAT_EQ(force(1, 0), -xForce);
    EXPECT_FLOAT_EQ(force(1, 1), -yForce);
    EXPECT_FLOAT_EQ(force(1, 2), real_t(0));

    EXPECT_FLOAT_EQ(force(2, 0), +xForce);
    EXPECT_FLOAT_EQ(force(2, 1), +yForce);
    EXPECT_FLOAT_EQ(force(2, 2), real_t(0));

    EXPECT_FLOAT_EQ(force(3, 0), +xForce);
    EXPECT_FLOAT_EQ(force(3, 1), -yForce);
    EXPECT_FLOAT_EQ(force(3, 2), real_t(0));
}

TEST_F(LJ_IdealGas_Test, AT)
{
    auto moleculesLambda = molecules.getModulatedLambda();
    Cabana::deep_copy(moleculesLambda, real_t(1));

    action::LJ_IdealGas LJ(cappingDistance, rc, sigma, epsilon, true);
    LJ.run(molecules, moleculesVerletList, atoms);

    constexpr auto xForce = real_t(0.22156665);
    constexpr auto yForce = real_t(1.3825009);

    data::HostAtoms h_atoms(atoms);
    auto force = h_atoms.getForce();

    EXPECT_FLOAT_EQ(force(0, 0), -xForce);
    EXPECT_FLOAT_EQ(force(0, 1), +yForce);
    EXPECT_FLOAT_EQ(force(0, 2), real_t(0));

    EXPECT_FLOAT_EQ(force(1, 0), -xForce);
    EXPECT_FLOAT_EQ(force(1, 1), -yForce);
    EXPECT_FLOAT_EQ(force(1, 2), real_t(0));

    EXPECT_FLOAT_EQ(force(2, 0), +xForce);
    EXPECT_FLOAT_EQ(force(2, 1), +yForce);
    EXPECT_FLOAT_EQ(force(2, 2), real_t(0));

    EXPECT_FLOAT_EQ(force(3, 0), +xForce);
    EXPECT_FLOAT_EQ(force(3, 1), -yForce);
    EXPECT_FLOAT_EQ(force(3, 2), real_t(0));
}

}  // namespace action
}  // namespace mrmd