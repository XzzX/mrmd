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
        data::Molecules molecules(2);

        auto pos = molecules.getPos();
        pos(0, 0) = -0.5_r;
        pos(0, 1) = 0_r;
        pos(0, 2) = 0_r;

        pos(1, 0) = +0.5_r;
        pos(1, 1) = 0_r;
        pos(1, 2) = 0_r;

        auto moleculesAtomsOffset = molecules.getAtomsOffset();
        auto moleculesNumAtoms = molecules.getNumAtoms();
        moleculesAtomsOffset(0) = 0;
        moleculesNumAtoms(0) = 2;
        moleculesAtomsOffset(1) = 2;
        moleculesNumAtoms(1) = 2;

        molecules.numLocalMolecules = 2;

        return molecules;
    }

    static auto getAtoms()
    {
        data::Atoms atoms(4);

        auto pos = atoms.getPos();
        pos(0, 0) = -0.5_r;
        pos(0, 1) = -0.5_r;
        pos(0, 2) = 0_r;
        atoms.getRelativeMass()(0) = 0.5_r;

        pos(1, 0) = -0.5_r;
        pos(1, 1) = +0.5_r;
        pos(1, 2) = 0_r;
        atoms.getRelativeMass()(1) = 0.5_r;

        pos(2, 0) = +0.5_r;
        pos(2, 1) = -0.5_r;
        pos(2, 2) = 0_r;
        atoms.getRelativeMass()(2) = 0.5_r;

        pos(3, 0) = +0.5_r;
        pos(3, 1) = +0.5_r;
        pos(3, 2) = 0_r;
        atoms.getRelativeMass()(3) = 0.5_r;

        atoms.numLocalAtoms = 4;

        return atoms;
    }

    void SetUp() override
    {
        molecules = getMolecules();

        auto cutoff = 2_r;
        auto cellRatio = 1_r;
        real_t minGrid[3] = {-1_r, -1_r, -1_r};
        real_t maxGrid[3] = {+1_r, +1_r, +1_r};
        auto expectedNumNeighbors = 4;
        moleculesVerletList.build(molecules.getPos(),
                                  0,
                                  molecules.numLocalMolecules,
                                  cutoff,
                                  cellRatio,
                                  minGrid,
                                  maxGrid,
                                  expectedNumNeighbors);

        atoms = getAtoms();
        atomsForce = atoms.getForce();
        Cabana::deep_copy(atomsForce, 0_r);
    }

    // void TearDown() override {}

    static constexpr real_t epsilon = 2_r;
    static constexpr real_t sigma = 0.9_r;
    static constexpr real_t rc = 2.5_r * sigma;
    static constexpr real_t cappingDistance = 0_r;
    static constexpr real_t eps = 0.001_r;

    data::Molecules molecules = data::Molecules(1);
    VerletList moleculesVerletList;
    data::Atoms atoms = data::Atoms(1);
    data::Atoms::force_t atomsForce;
};

TEST_F(LJ_IdealGas_Test, CG)
{
    Cabana::deep_copy(atomsForce, 2_r);

    auto moleculesLambda = molecules.getModulatedLambda();
    Cabana::deep_copy(moleculesLambda, 0_r);
    action::LJ_IdealGas LJ(cappingDistance, rc, sigma, epsilon, true);
    LJ.run(molecules, moleculesVerletList, atoms);

    for (idx_t idx = 0; idx < 4; ++idx)
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            EXPECT_FLOAT_EQ(atomsForce(idx, dim), 2_r);
        }
    }
}

TEST_F(LJ_IdealGas_Test, HY)
{
    auto moleculesLambda = molecules.getModulatedLambda();
    Cabana::deep_copy(moleculesLambda, 0.5_r);

    action::LJ_IdealGas LJ(cappingDistance, rc, sigma, epsilon, true);
    LJ.run(molecules, moleculesVerletList, atoms);

    constexpr auto xForce = 0.22156665_r * 0.5_r;
    constexpr auto yForce = 1.3825009_r * 0.5_r;

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), -xForce);
    EXPECT_FLOAT_EQ(force(0, 1), +yForce);
    EXPECT_FLOAT_EQ(force(0, 2), 0_r);

    EXPECT_FLOAT_EQ(force(1, 0), -xForce);
    EXPECT_FLOAT_EQ(force(1, 1), -yForce);
    EXPECT_FLOAT_EQ(force(1, 2), 0_r);

    EXPECT_FLOAT_EQ(force(2, 0), +xForce);
    EXPECT_FLOAT_EQ(force(2, 1), +yForce);
    EXPECT_FLOAT_EQ(force(2, 2), 0_r);

    EXPECT_FLOAT_EQ(force(3, 0), +xForce);
    EXPECT_FLOAT_EQ(force(3, 1), -yForce);
    EXPECT_FLOAT_EQ(force(3, 2), 0_r);
}

TEST_F(LJ_IdealGas_Test, AT)
{
    auto moleculesLambda = molecules.getModulatedLambda();
    Cabana::deep_copy(moleculesLambda, 1_r);

    action::LJ_IdealGas LJ(cappingDistance, rc, sigma, epsilon, true);
    LJ.run(molecules, moleculesVerletList, atoms);

    constexpr auto xForce = 0.22156665_r;
    constexpr auto yForce = 1.3825009_r;

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto force = Cabana::slice<data::Atoms::FORCE>(hAoSoA);

    EXPECT_FLOAT_EQ(force(0, 0), -xForce);
    EXPECT_FLOAT_EQ(force(0, 1), +yForce);
    EXPECT_FLOAT_EQ(force(0, 2), 0_r);

    EXPECT_FLOAT_EQ(force(1, 0), -xForce);
    EXPECT_FLOAT_EQ(force(1, 1), -yForce);
    EXPECT_FLOAT_EQ(force(1, 2), 0_r);

    EXPECT_FLOAT_EQ(force(2, 0), +xForce);
    EXPECT_FLOAT_EQ(force(2, 1), +yForce);
    EXPECT_FLOAT_EQ(force(2, 2), 0_r);

    EXPECT_FLOAT_EQ(force(3, 0), +xForce);
    EXPECT_FLOAT_EQ(force(3, 1), -yForce);
    EXPECT_FLOAT_EQ(force(3, 2), 0_r);
}

}  // namespace action
}  // namespace mrmd