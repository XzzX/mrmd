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

#include "GhostExchange.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
size_t countWithinCutoff(const data::Atoms& atoms,
                         const real_t cutoff,
                         const double* box,
                         const bool periodic)
{
    auto numLocalAtoms = atoms.numLocalAtoms;
    auto numGhostAtoms = atoms.numGhostAtoms;
    auto rcSqr = cutoff * cutoff;
    auto pos = atoms.getPos();
    real_t box_diameter[3] = {box[0], box[1], box[2]};

    size_t count = 0;
    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx, size_t& sum)
    {
        for (auto jdx = idx + 1; jdx < numLocalAtoms + numGhostAtoms; ++jdx)
        {
            auto dx = std::abs(pos(idx, 0) - pos(jdx, 0));
            if (periodic && (dx > box_diameter[0] * 0.5_r)) dx -= box_diameter[0];
            auto dy = std::abs(pos(idx, 1) - pos(jdx, 1));
            if (periodic && (dy > box_diameter[1] * 0.5_r)) dy -= box_diameter[1];
            auto dz = std::abs(pos(idx, 2) - pos(jdx, 2));
            if (periodic && (dz > box_diameter[2] * 0.5_r)) dz -= box_diameter[2];
            auto distSqr = dx * dx + dy * dy + dz * dz;
            if (distSqr < rcSqr)
            {
                ++sum;
            }
        }
    };
    Kokkos::parallel_reduce(policy, kernel, count);

    return count;
}

class GhostExchangeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto pos = h_atoms.getPos();
        int64_t idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    pos(idx, 0) = x;
                    pos(idx, 1) = y;
                    pos(idx, 2) = z;
                    ++idx;
                }
        EXPECT_EQ(idx, 27);
        h_atoms.numLocalAtoms = 27;
        h_atoms.numGhostAtoms = 0;
        h_atoms.resize(h_atoms.numLocalAtoms + h_atoms.numGhostAtoms);
        data::deep_copy(atoms, h_atoms);
    }

    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    data::HostAtoms h_atoms = data::HostAtoms(200);
    data::Atoms atoms = data::Atoms(200);
};

void selfExchange(const data::Subdomain& subdomain, data::Atoms& atoms, const idx_t& dimension)
{
    EXPECT_EQ(atoms.numGhostAtoms, 0);
    auto ghostExchange = GhostExchange();
    ghostExchange.resetCorrespondingRealAtoms(atoms);
    auto correspondingRealAtom = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), ghostExchange.createGhostAtoms(atoms, subdomain, dimension));
    EXPECT_EQ(atoms.numGhostAtoms, 18);
    for (auto idx = 0; idx < atoms.numLocalAtoms + atoms.numGhostAtoms; ++idx)
    {
        if (idx < atoms.numLocalAtoms)
        {
            EXPECT_EQ(correspondingRealAtom(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealAtom(idx), atoms.numLocalAtoms);
        }
    }
}
TEST_F(GhostExchangeTest, SelfExchangeX) { selfExchange(subdomain, atoms, COORD_X); }
TEST_F(GhostExchangeTest, SelfExchangeY) { selfExchange(subdomain, atoms, COORD_Y); }
TEST_F(GhostExchangeTest, SelfExchangeZ) { selfExchange(subdomain, atoms, COORD_Z); }

TEST_F(GhostExchangeTest, SelfExchangeXYZ)
{
    auto ghostExchange = GhostExchange();
    auto correspondingRealAtom = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), ghostExchange.createGhostAtomsXYZ(atoms, subdomain));
    EXPECT_EQ(atoms.numGhostAtoms, 98);
    for (auto idx = 0; idx < atoms.numLocalAtoms + atoms.numGhostAtoms; ++idx)
    {
        if (idx < atoms.numLocalAtoms)
        {
            EXPECT_EQ(correspondingRealAtom(idx), -1);
        }
        else
        {
            EXPECT_LT(correspondingRealAtom(idx), atoms.numLocalAtoms);
        }
    }
}

TEST_F(GhostExchangeTest, CountPairs)
{
    size_t numPairs = 0;
    numPairs = countWithinCutoff(atoms, 1.1_r, subdomain.diameter.data(), false);
    EXPECT_EQ(numPairs, 12 * 3 + 9 * 2);

    numPairs = countWithinCutoff(atoms, 1.1_r, subdomain.diameter.data(), true);
    EXPECT_EQ(numPairs, 27 * 6 / 2);

    auto ghostExchange = GhostExchange();
    ghostExchange.createGhostAtomsXYZ(atoms, subdomain);
    EXPECT_EQ(atoms.numGhostAtoms, 98);

    numPairs = countWithinCutoff(atoms, 1.1_r, subdomain.diameter.data(), false);
    EXPECT_EQ(numPairs, 108);
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd