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

#include "MeanSquareDisplacement.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace analysis
{
using MeanSquareDisplacementTest = test::SingleAtom;

TEST_F(MeanSquareDisplacementTest, no_displacement)
{
    data::Subdomain subdomain({0_r, 0_r, 0_r}, {10_r, 10_r, 10_r}, 1_r);
    analysis::MeanSquareDisplacement msd;
    msd.reset(atoms);
    auto meanSqDisplacement = msd.calc(atoms, subdomain);
    EXPECT_FLOAT_EQ(meanSqDisplacement, 0_r);
}
}  // namespace analysis
}  // namespace mrmd