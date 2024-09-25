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

#include <gtest/gtest.h>

#include "action/Shake.hpp"
#include "action/VelocityVerlet.hpp"
#include "test/DiamondFixture.hpp"

namespace mrmd
{
using ConstraintTest = test::DiamondFixture;

TEST_F(ConstraintTest, ShakeVelocityVerlet)
{
    auto dt = 0.1_r;

    data::BondView::host_mirror_type bonds("bonds", 1);
    bonds(0).idx = 0;
    bonds(0).jdx = 1;
    bonds(0).eqDistance = 1_r;

    action::MoleculeConstraints mc(1, 1);
    mc.setConstraints(bonds);

    mc.enforcePositionalConstraints(molecules, atoms, dt);
    action::VelocityVerlet::preForceIntegrate(atoms, dt);
    auto atomsForce = atoms.getForce();
    Cabana::deep_copy(atomsForce, 0_r);
    action::VelocityVerlet::postForceIntegrate(atoms, dt);
    mc.enforceVelocityConstraints(molecules, atoms, dt);

    auto hAoSoA = Cabana::create_mirror_view_and_copy(Kokkos::HostSpace(), atoms.getAoSoA());
    auto pos = Cabana::slice<data::Atoms::POS>(hAoSoA);
    auto vel = Cabana::slice<data::Atoms::VEL>(hAoSoA);
    auto calcDist = [=](idx_t idx, idx_t jdx)
    {
        real_t dx[3];
        dx[0] = pos(idx, 0) - pos(jdx, 0);
        dx[1] = pos(idx, 1) - pos(jdx, 1);
        dx[2] = pos(idx, 2) - pos(jdx, 2);
        return std::sqrt(util::dot3(dx, dx));
    };
    auto calcRelVel = [=](idx_t idx, idx_t jdx)
    {
        real_t dx[3];
        dx[0] = pos(idx, 0) - pos(jdx, 0);
        dx[1] = pos(idx, 1) - pos(jdx, 1);
        dx[2] = pos(idx, 2) - pos(jdx, 2);

        real_t dv[3];
        dv[0] = vel(idx, 0) - vel(jdx, 0);
        dv[1] = vel(idx, 1) - vel(jdx, 1);
        dv[2] = vel(idx, 2) - vel(jdx, 2);

        return util::dot3(dx, dv) / std::sqrt(util::dot3(dx, dx));
    };
    EXPECT_FLOAT_EQ(calcDist(0, 1), 1_r);
    EXPECT_FLOAT_EQ(calcRelVel(0, 1) + 1_r, 1_r);
}

}  // namespace mrmd

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    return RUN_ALL_TESTS();
}
