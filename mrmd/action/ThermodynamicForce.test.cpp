// Copyright 2024 Sebastian Eibl
// Copyright 2026 Julian Friedrich Hille
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

#include "ThermodynamicForce.hpp"

#include <cmath>

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace action
{
struct TestPredicate
{
    real_t lower;
    real_t upper;

    KOKKOS_INLINE_FUNCTION bool operator()(real_t x, real_t, real_t) const
    {
        return x > lower && x < upper;
    }
};

data::Atoms getAtoms()
{
    idx_t numAtoms = 100;
    data::Atoms atoms(numAtoms);
    atoms.numLocalAtoms = numAtoms;
    auto pos = atoms.getPos();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx) { pos(idx, 0) = real_c(idx) / 10_r; };
    Kokkos::parallel_for("getAtoms", policy, kernel);
    Kokkos::fence();
    return atoms;
}

ThermodynamicForce getThermodynamicForce()
{
    data::Subdomain subdomain({0_r, 0_r, 0_r}, {10_r, 10_r, 10_r}, 1_r);
    ThermodynamicForce thermodynamicForce({1_r}, subdomain, 1_r, {1_r});

    auto forceHistogram = thermodynamicForce.getForce();
    auto policyForce = Kokkos::RangePolicy<>(0, forceHistogram.numBins);
    auto kernelForce = KOKKOS_LAMBDA(const idx_t& idx)
    {
        forceHistogram.data(idx, 0) = real_c(idx);
    };
    Kokkos::parallel_for("getThermodynamicForce", policyForce, kernelForce);
    Kokkos::fence();
    thermodynamicForce.setForce(forceHistogram.data);
    return thermodynamicForce;
}

TEST(ThermodynamicForce, apply)
{
    auto atoms = getAtoms();
    auto thermodynamicForce = getThermodynamicForce();

    thermodynamicForce.apply(atoms);

    data::HostAtoms h_atoms(atoms);
    auto h_pos = h_atoms.getPos();
    auto h_force = h_atoms.getForce();
    for (idx_t idx = 0; idx < h_atoms.numLocalAtoms; ++idx)
    {
        EXPECT_FLOAT_EQ(h_force(idx, 0), std::floor(h_pos(idx, 0)));
    }
}

TEST(ThermodynamicForce, apply_if)
{
    auto atoms = getAtoms();
    auto thermodynamicForce = getThermodynamicForce();

    thermodynamicForce.apply_if(atoms, TestPredicate{0.5_r, 4.5_r});

    data::HostAtoms h_atoms(atoms);
    auto h_pos = h_atoms.getPos();
    auto h_force = h_atoms.getForce();
    for (idx_t idx = 0; idx < h_atoms.numLocalAtoms; ++idx)
    {
        if (h_pos(idx, 0) > 0.5_r && h_pos(idx, 0) < 4.5_r)
        {
            EXPECT_FLOAT_EQ(h_force(idx, 0), std::floor(h_pos(idx, 0)));
        }
        else  // forces should not be updated due to predicate
        {
            EXPECT_FLOAT_EQ(h_force(idx, 0), 0_r);
        }
    }
}

TEST(ThermodynamicForce, applyInterpolated_if)
{
    auto atoms = getAtoms();
    auto thermodynamicForce = getThermodynamicForce();

    thermodynamicForce.applyInterpolated_if(atoms, TestPredicate{0.5_r, 4.5_r});

    data::HostAtoms h_atoms(atoms);
    auto h_pos = h_atoms.getPos();
    auto h_force = h_atoms.getForce();
    for (idx_t idx = 0; idx < h_atoms.numLocalAtoms; ++idx)
    {
        if (h_pos(idx, 0) > 0.5_r && h_pos(idx, 0) < 4.5_r)
        {
            EXPECT_FLOAT_EQ(h_force(idx, 0), h_pos(idx, 0) - 0.5_r);
        }
        else  // forces should not be updated due to predicate
        {
            EXPECT_FLOAT_EQ(h_force(idx, 0), 0_r);
        }
    }
}

}  // namespace action
}  // namespace mrmd