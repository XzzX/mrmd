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
    const data::Subdomain subdomain_;
    IndexView correspondingRealAtom_;
    impl::MultiResPeriodicGhostExchange ghostExchange_;
    impl::UpdateGhostAtoms updateGhostAtoms_;
    impl::AccumulateForce accumulateForce_;

public:
    void exchangeRealAtoms(data::Molecules& molecules, data::Atoms& atoms)
    {
        realAtomsExchange(subdomain_, molecules, atoms);
    }

    void createGhostAtoms(data::Molecules& molecules, data::Atoms& atoms)
    {
        correspondingRealAtom_ = ghostExchange_.createGhostAtomsXYZ(molecules, atoms);
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

    MultiResGhostLayer(const data::Subdomain& subdomain)
        : subdomain_(subdomain), ghostExchange_(subdomain), updateGhostAtoms_(subdomain)
    {
    }
};

}  // namespace communication
}  // namespace mrmd