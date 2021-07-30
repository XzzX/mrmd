#include "LJ_IdealGas.hpp"

#include <gtest/gtest.h>

#include "data/Molecules.hpp"
#include "data/Particles.hpp"

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

        auto moleculesAtomsEndIdx = molecules.getAtomsEndIdx();
        moleculesAtomsEndIdx(0) = 2;
        moleculesAtomsEndIdx(1) = 4;

        molecules.numLocalMolecules = 2;

        return molecules;
    }

    static auto getAtoms()
    {
        data::Particles atoms(4);

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

        atoms.numLocalParticles = 4;

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
    data::Particles atoms = data::Particles(1);
    data::Particles::force_t atomsForce;
};

TEST_F(LJ_IdealGas_Test, CG)
{
    Cabana::deep_copy(atomsForce, 2_r);

    auto moleculesLambda = molecules.getLambda();
    Cabana::deep_copy(moleculesLambda, 0_r);
    action::LJ_IdealGas::applyForces(
        cappingDistance, rc, sigma, epsilon, molecules, moleculesVerletList, atoms);

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
    auto moleculesLambda = molecules.getLambda();
    Cabana::deep_copy(moleculesLambda, 0.5_r);

    action::LJ_IdealGas::applyForces(
        cappingDistance, rc, sigma, epsilon, molecules, moleculesVerletList, atoms);

    constexpr auto xForce = 0.22156665_r * 0.5_r;
    constexpr auto yForce = 1.3825009_r * 0.5_r;

    EXPECT_FLOAT_EQ(atomsForce(0, 0), -xForce);
    EXPECT_FLOAT_EQ(atomsForce(0, 1), +yForce);
    EXPECT_FLOAT_EQ(atomsForce(0, 2), 0_r);

    EXPECT_FLOAT_EQ(atomsForce(1, 0), -xForce);
    EXPECT_FLOAT_EQ(atomsForce(1, 1), -yForce);
    EXPECT_FLOAT_EQ(atomsForce(1, 2), 0_r);

    EXPECT_FLOAT_EQ(atomsForce(2, 0), +xForce);
    EXPECT_FLOAT_EQ(atomsForce(2, 1), +yForce);
    EXPECT_FLOAT_EQ(atomsForce(2, 2), 0_r);

    EXPECT_FLOAT_EQ(atomsForce(3, 0), +xForce);
    EXPECT_FLOAT_EQ(atomsForce(3, 1), -yForce);
    EXPECT_FLOAT_EQ(atomsForce(3, 2), 0_r);
}

TEST_F(LJ_IdealGas_Test, AT)
{
    auto moleculesLambda = molecules.getLambda();
    Cabana::deep_copy(moleculesLambda, 1_r);

    action::LJ_IdealGas::applyForces(
        cappingDistance, rc, sigma, epsilon, molecules, moleculesVerletList, atoms);

    constexpr auto xForce = 0.22156665_r;
    constexpr auto yForce = 1.3825009_r;

    EXPECT_FLOAT_EQ(atomsForce(0, 0), -xForce);
    EXPECT_FLOAT_EQ(atomsForce(0, 1), +yForce);
    EXPECT_FLOAT_EQ(atomsForce(0, 2), 0_r);

    EXPECT_FLOAT_EQ(atomsForce(1, 0), -xForce);
    EXPECT_FLOAT_EQ(atomsForce(1, 1), -yForce);
    EXPECT_FLOAT_EQ(atomsForce(1, 2), 0_r);

    EXPECT_FLOAT_EQ(atomsForce(2, 0), +xForce);
    EXPECT_FLOAT_EQ(atomsForce(2, 1), +yForce);
    EXPECT_FLOAT_EQ(atomsForce(2, 2), 0_r);

    EXPECT_FLOAT_EQ(atomsForce(3, 0), +xForce);
    EXPECT_FLOAT_EQ(atomsForce(3, 1), -yForce);
    EXPECT_FLOAT_EQ(atomsForce(3, 2), 0_r);
}

}  // namespace action
}  // namespace mrmd