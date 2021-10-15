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

    void createGhostAtoms(data::Atoms& atoms)
    {
        correspondingRealAtom_ = ghostExchange_.createGhostAtomsXYZ(atoms);
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

    GhostLayer(const data::Subdomain& subdomain) : ghostExchange_(subdomain) {}
};

}  // namespace communication
}  // namespace mrmd