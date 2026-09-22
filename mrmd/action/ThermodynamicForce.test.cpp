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

#include <gtest/gtest.h>

#include <cmath>

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

    KOKKOS_INLINE_FUNCTION bool operator()(real_t x) const { return x > lower && x < upper; }
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

data::Atoms getAtomsNonuniform()
{
    data::HostAtoms h_atoms(45);
    auto h_pos = h_atoms.getPos();

    idx_t numAtoms = 0;
    for (idx_t idx = 0; idx < 10; ++idx)
    {
        for (idx_t jdx = 0; jdx < idx; ++jdx)
        {
            h_pos(numAtoms, 0) = real_c(idx) + 0.5_r;
            ++numAtoms;
        }
    }
    assert(numAtoms == 45);
    h_atoms.numLocalAtoms = 45;

    data::Atoms atoms(h_atoms);
    data::deep_copy(atoms, h_atoms);

    return atoms;
}

ThermodynamicForce getThermodynamicForce()
{
    data::Subdomain subdomain({0_r, 0_r, 0_r}, {10_r, 1_r, 1_r}, 1_r);
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

TEST(ThermodynamicForce, getGrid)
{
    auto thermodynamicForce = getThermodynamicForce();
    auto h_grid = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), createGrid(thermodynamicForce.getDensityProfile()));

    for (idx_t idx = 0; idx < h_grid.extent(0); ++idx)
    {
        EXPECT_FLOAT_EQ(h_grid(idx), real_c(idx) + 0.5_r);
    };
}

TEST(ThermodynamicForce, sample)
{
    auto atoms = getAtoms();
    auto thermodynamicForce = getThermodynamicForce();

    thermodynamicForce.sample(atoms);

    auto h_densityProfile = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), thermodynamicForce.getDensityProfile(0));
    for (idx_t idx = 0; idx < h_densityProfile.extent(0); ++idx)
    {
        EXPECT_FLOAT_EQ(h_densityProfile(idx), 10_r);
    };
}

TEST(ThermodynamicForce, sampleNonuniform)
{
    auto atoms = getAtomsNonuniform();
    auto thermodynamicForce = getThermodynamicForce();

    thermodynamicForce.sample(atoms);

    auto h_densityProfile = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), thermodynamicForce.getDensityProfile(0));
    for (idx_t idx = 0; idx < h_densityProfile.extent(0); ++idx)
    {
        EXPECT_FLOAT_EQ(h_densityProfile(idx), real_c(idx));
    };
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

TEST(ThermodynamicForce, update)
{
    auto atoms = getAtomsNonuniform();
    auto thermodynamicForce = getThermodynamicForce();

    thermodynamicForce.sample(atoms);
    thermodynamicForce.update(1_r, 0_r);

    auto forceHistogram = thermodynamicForce.getForce();

    auto h_force =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), thermodynamicForce.getForce(0));
    for (idx_t idx = 0; idx < forceHistogram.numBins; ++idx)
    {
        EXPECT_FLOAT_EQ(h_force(idx), real_c(idx) - 1_r);
    };
}

TEST(ThermodynamicForce, update_if)
{
    auto atoms = getAtomsNonuniform();
    auto thermodynamicForce = getThermodynamicForce();

    thermodynamicForce.sample(atoms);

    thermodynamicForce.update_if(1_r, 0_r, TestPredicate{0.51_r, 4.49_r});

    auto forceHistogram = thermodynamicForce.getForce();

    auto h_force =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), thermodynamicForce.getForce(0));
    auto h_grid =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), createGrid(forceHistogram));
    for (idx_t idx = 0; idx < forceHistogram.numBins; ++idx)
    {
        if (h_grid(idx) > 0.51_r &&
            h_grid(idx) < 4.49_r)  // only bins corresponding to the predicate should be updated
        {
            EXPECT_FLOAT_EQ(h_force(idx), real_c(idx) - 1_r);
        }
        else
        {
            EXPECT_FLOAT_EQ(h_force(idx), real_c(idx));
        }
    };
}
}  // namespace action
}  // namespace mrmd