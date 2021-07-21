#pragma once

#include "communication/AccumulateForce.hpp"
#include "communication/MultiResPeriodicGhostExchange.hpp"
#include "communication/MultiResRealParticlesExchange.hpp"
#include "communication/UpdateGhostParticles.hpp"
#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
class MultiResGhostLayer
{
private:
    const data::Subdomain subdomain_;
    IndexView correspondingRealParticle_;
    impl::MultiResPeriodicGhostExchange ghostExchange_;
    impl::UpdateGhostParticles updateGhostParticles_;
    impl::AccumulateForce accumulateForce_;

public:
    void exchangeRealParticles(data::Molecules& molecules, data::Particles& atoms)
    {
        realParticlesExchange(subdomain_, molecules, atoms);
    }

    void createGhostParticles(data::Molecules& molecules, data::Particles& atoms)
    {
        correspondingRealParticle_ = ghostExchange_.createGhostParticlesXYZ(molecules, atoms);
    }

    void updateGhostParticles(data::Particles& atoms)
    {
        assert(correspondingRealParticle_.extent(0) >= atoms.size());

        updateGhostParticles_.updateOnlyPos(atoms, correspondingRealParticle_);
    }

    void contributeBackGhostToReal(data::Particles& atoms)
    {
        accumulateForce_.ghostToReal(atoms, correspondingRealParticle_);
    }

    MultiResGhostLayer(const data::Subdomain& subdomain)
        : subdomain_(subdomain), ghostExchange_(subdomain), updateGhostParticles_(subdomain)
    {
    }
};

}  // namespace communication
}  // namespace mrmd