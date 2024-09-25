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

#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "util/Kokkos_grow.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
struct DoubleCounter
{
    idx_t atoms = 0;
    idx_t molecules = 0;

    KOKKOS_INLINE_FUNCTION
    DoubleCounter() = default;
    KOKKOS_INLINE_FUNCTION
    DoubleCounter(idx_t newAtoms, idx_t newMolecules) : atoms(newAtoms), molecules(newMolecules) {}
    KOKKOS_INLINE_FUNCTION
    DoubleCounter(const DoubleCounter& rhs) = default;

    KOKKOS_INLINE_FUNCTION
    DoubleCounter(const volatile DoubleCounter& rhs)
    {
        atoms = rhs.atoms;
        molecules = rhs.molecules;
    }

    KOKKOS_INLINE_FUNCTION
    void operator=(const DoubleCounter& rhs) volatile
    {
        atoms = rhs.atoms;
        molecules = rhs.molecules;
    }

    KOKKOS_INLINE_FUNCTION
    void operator=(volatile const DoubleCounter& rhs) volatile
    {
        atoms = rhs.atoms;
        molecules = rhs.molecules;
    }

    KOKKOS_INLINE_FUNCTION
    DoubleCounter& operator+=(const DoubleCounter& rhs)
    {
        atoms += rhs.atoms;
        molecules += rhs.molecules;
        return *this;
    }
};

KOKKOS_INLINE_FUNCTION
DoubleCounter operator+(const DoubleCounter& lhs, const DoubleCounter& rhs)
{
    return DoubleCounter(lhs.atoms + rhs.atoms, lhs.molecules + rhs.molecules);
}

class MultiResPeriodicGhostExchange
{
private:
    struct Info
    {
        enum
        {
            POSITIVE_MOLECULE_IDX = 0,
            POSITIVE_NUM_ATOMS = 1,
            NEGATIVE_MOLECULE_IDX = 2,
            NEGATIVE_NUM_ATOMS = 3
        };
    };
    /// Selected indices to be communicated in a certain direction.
    /// first dim: index
    /// second dim: direction
    Kokkos::View<idx_t* [4]> communicationInfo_;

    struct Item
    {
        enum
        {
            POSITIVE_MOLECULES = 0,
            POSITIVE_ATOMS = 1,
            NEGATIVE_MOLECULES = 2,
            NEGATIVE_ATOMS = 3
        };
    };
    /// number of atoms to communicate in each direction
    Kokkos::View<idx_t[4]> numberOfCommunicationItems_;

    /// Stores the corresponding real atom index for every ghost atom.
    IndexView atomsCorrespondingRealAtom_;

    /// Stores the corresponding real atom index for every ghost atom.
    IndexView moleculesCorrespondingRealMolecule_;

public:
    IndexView createGhostAtoms(data::Molecules& molecules,
                               data::Atoms& atoms,
                               const data::Subdomain& subdomain,
                               const idx_t& dim);

    IndexView createGhostAtomsXYZ(data::Molecules& molecules,
                                  data::Atoms& atoms,
                                  const data::Subdomain& subdomain);

    void resetCorrespondingRealAtoms(data::Atoms& atoms);
    void resetCorrespondingRealMolecules(data::Molecules& molecules);

    explicit MultiResPeriodicGhostExchange(const idx_t& initialSize = 0);
};
}  // namespace impl
}  // namespace communication
}  // namespace mrmd
