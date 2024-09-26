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

namespace mrmd
{
namespace communication
{
/**
 * Intended for single process use. Maps atoms that went
 * out of the domain back in a periodic fashion. Mapping is
 * done based on molecule position and atoms are moved according
 * to their molecule.
 *
 * @pre Atom position is at most one periodic copy away
 * from the subdomain.
 * @post Atom position lies within half-open interval
 * [min, max) for all coordinate dimensions.
 */
void realAtomsExchange(const data::Subdomain& subdomain,
                       const data::Molecules& molecules,
                       const data::Atoms& atoms);

}  // namespace communication
}  // namespace mrmd