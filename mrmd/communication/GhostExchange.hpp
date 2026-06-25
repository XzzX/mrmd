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
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "util/Kokkos_grow.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class GhostExchange
{
private:
    /// Selected indices to be communicated in a certain direction.
    /// first dim: index
    /// second dim: direction
    Kokkos::View<idx_t* [2]> atomsToCommunicateAll_;

    /// number of atoms to communicate in each direction
    Kokkos::View<idx_t[2]> numberOfAtomsToCommunicate_;

    /// Stores the corresponding real atom index for every ghost atom.
    IndexView correspondingRealAtom_;

public:
    IndexView createGhostAtoms(data::Atoms& atoms,
                               const data::Subdomain& subdomain,
                               const AXIS& axis);

    IndexView createGhostAtomsXYZ(data::Atoms& atoms, const data::Subdomain& subdomain);

    void resetCorrespondingRealAtoms(data::Atoms& atoms);

    GhostExchange(const idx_t& initialSize = 100);
};

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
