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

#include "UpdateMolecules.hpp"

#include <gtest/gtest.h>

#include "test/DiamondFixture.hpp"

namespace mrmd
{
namespace action
{
using UpdateMoleculesTest = test::DiamondFixture;

struct Weight
{
    KOKKOS_INLINE_FUNCTION
    void operator()(const real_t x,
                    const real_t y,
                    const real_t z,
                    real_t& lambda,
                    real_t& modulatedLambda,
                    real_t& gradLambdaX,
                    real_t& gradLambdaY,
                    real_t& gradLambdaZ) const
    {
        modulatedLambda = x > 0 ? 0.7_r : -0.7_r;
    }
};

TEST_F(UpdateMoleculesTest, update)
{
    Weight weight;
    action::UpdateMolecules::update(molecules, atoms, weight);

    data::HostMolecules h_molecules(molecules);
    EXPECT_FLOAT_EQ(h_molecules.getPos()(0, 0), 0.25_r);
    EXPECT_FLOAT_EQ(h_molecules.getPos()(0, 1), 0.75_r);
    EXPECT_FLOAT_EQ(h_molecules.getPos()(0, 2), 0_r);

    EXPECT_FLOAT_EQ(h_molecules.getModulatedLambda()(0), 0.7_r);

    EXPECT_FLOAT_EQ(h_molecules.getPos()(1, 0), -0.25_r);
    EXPECT_FLOAT_EQ(h_molecules.getPos()(1, 1), -0.75_r);
    EXPECT_FLOAT_EQ(h_molecules.getPos()(1, 2), 0_r);

    EXPECT_FLOAT_EQ(h_molecules.getModulatedLambda()(1), -0.7_r);
}

}  // namespace action
}  // namespace mrmd