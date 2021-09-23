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
    const data::Subdomain subdomain_;
    IndexView correspondingRealAtom_;
    impl::PeriodicMapping periodicMapping_;
    impl::GhostExchange ghostExchange_;
    impl::UpdateGhostAtoms updateGhostAtoms_;
    impl::AccumulateForce accumulateForce_;

public:
    void exchangeRealAtoms(data::Atoms& atoms) { periodicMapping_.mapIntoDomain(atoms); }

    void createGhostAtoms(data::Atoms& atoms)
    {
        correspondingRealAtom_ = ghostExchange_.createGhostAtomsXYZ(atoms);
    }

    void updateGhostAtoms(data::Atoms& atoms)
    {
        assert(correspondingRealAtom_.extent(0) >= atoms.size());

        updateGhostAtoms_.updateOnlyPos(atoms, correspondingRealAtom_);
    }

    void contributeBackGhostToReal(data::Atoms& atoms)
    {
        accumulateForce_.ghostToReal(atoms, correspondingRealAtom_);
    }

    GhostLayer(const data::Subdomain& subdomain)
        : subdomain_(subdomain),
          periodicMapping_(subdomain),
          ghostExchange_(subdomain),
          updateGhostAtoms_(subdomain)
    {
    }
};

}  // namespace communication
}  // namespace mrmd