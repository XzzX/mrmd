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

#include "communication/AccumulateForce.hpp"
#include "communication/MultiResPeriodicGhostExchange.hpp"
#include "communication/MultiResRealAtomsExchange.hpp"
#include "communication/UpdateGhostAtoms.hpp"
#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
class MultiResGhostLayer
{
private:
    IndexView correspondingRealAtom_;
    impl::MultiResPeriodicGhostExchange ghostExchange_;

public:
    void exchangeRealAtoms(data::Molecules& molecules,
                           data::Atoms& atoms,
                           const data::Subdomain& subdomain)
    {
        realAtomsExchange(subdomain, molecules, atoms);
    }

    void createGhostAtoms(data::Molecules& molecules,
                          data::Atoms& atoms,
                          const data::Subdomain& subdomain)
    {
        correspondingRealAtom_ = ghostExchange_.createGhostAtomsXYZ(molecules, atoms, subdomain);
    }

    void updateGhostAtoms(data::Atoms& atoms, const data::Subdomain& subdomain)
    {
        assert(correspondingRealAtom_.extent(0) >= atoms.size());

        impl::UpdateGhostAtoms::updateOnlyPos(atoms, correspondingRealAtom_, subdomain);
    }

    void contributeBackGhostToReal(data::Atoms& atoms)
    {
        impl::AccumulateForce::ghostToReal(atoms, correspondingRealAtom_);
    }
};

}  // namespace communication
}  // namespace mrmd