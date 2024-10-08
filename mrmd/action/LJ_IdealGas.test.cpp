// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
        pos(0, 0) = -0.5_r;
        pos(0, 1) = 0_r;
        pos(0, 2) = 0_r;

        pos(1, 0) = +0.5_r;
        pos(1, 1) = 0_r;
        pos(1, 2) = 0_r;

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

        pos(0, 0) = -0.5_r;
        pos(0, 1) = -0.5_r;
        pos(0, 2) = 0_r;
        relativeMass(0) = 0.5_r;

        pos(1, 0) = -0.5_r;
        pos(1, 1) = +0.5_r;
        pos(1, 2) = 0_r;
        relativeMass(1) = 0.5_r;

        pos(2, 0) = +0.5_r;
        pos(2, 1) = -0.5_r;
        pos(2, 2) = 0_r;
        relativeMass(2) = 0.5_r;

        pos(3, 0) = +0.5_r;
        pos(3, 1) = +0.5_r;
        pos(3, 2) = 0_r;
        relativeMass(3) = 0.5_r;

        Cabana::deep_copy(atoms.getAoSoA(), hAtoms);

        auto type = atoms.getType();
        Cabana::deep_copy(type, 0);

        atoms.numLocalAtoms = 4;

        return atoms;
    }

    void SetUp() override
    {
        data::deep_copy(molecules, getMolecules());

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

        data::deep_copy(atoms, getAtoms());
        auto atomsForce = atoms.getForce();
        Cabana::deep_copy(atomsForce, 0_r);
    }

    // void TearDown() override {}

    static constexpr real_t epsilon = 2_r;
    static constexpr real_t sigma = 0.9_r;
    static constexpr real_t rc = 2.5_r * sigma;
    static constexpr real_t cappingDistance = 0_r;
    static constexpr real_t eps = 0.001_r;

    data::Molecules molecules = data::Molecules(1);
    HalfVerletList moleculesVerletList;
    data::Atoms atoms = data::Atoms(1);
};

TEST_F(LJ_IdealGas_Test, CG)
{
    auto atomsForce = atoms.getForce();
    Cabana::deep_copy(atomsForce, 2_r);

    auto moleculesLambda = molecules.getModulatedLambda();
    Cabana::deep_copy(moleculesLambda, 0_r);
    action::LJ_IdealGas LJ(cappingDistance, rc, sigma, epsilon, true);
    LJ.run(molecules, moleculesVerletList, atoms);

    data::HostAtoms h_atoms(atoms);
    for (idx_t idx = 0; idx < 4; ++idx)
    {
        for (auto dim = 0; dim < 3; ++dim)
        {
            EXPECT_FLOAT_EQ(h_atoms.getForce()(idx, dim), 2_r);
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

    data::HostAtoms h_atoms(atoms);
    auto force = h_atoms.getForce();

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

    data::HostAtoms h_atoms(atoms);
    auto force = h_atoms.getForce();

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