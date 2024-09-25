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
#include "communication/GhostExchange.hpp"
#include "communication/PeriodicMapping.hpp"
#include "communication/UpdateGhostAtoms.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
class GhostLayer
{
private:
    IndexView correspondingRealAtom_;
    impl::GhostExchange ghostExchange_;

public:
    void exchangeRealAtoms(data::Atoms& atoms, const data::Subdomain& subdomain)
    {
        impl::PeriodicMapping::mapIntoDomain(atoms, subdomain);
    }

    void createGhostAtoms(data::Atoms& atoms, const data::Subdomain& subdomain)
    {
        correspondingRealAtom_ = ghostExchange_.createGhostAtomsXYZ(atoms, subdomain);
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