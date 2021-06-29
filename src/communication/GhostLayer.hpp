#pragma once

#include "communication/AccumulateForce.hpp"
#include "communication/GhostExchange.hpp"
#include "communication/PeriodicMapping.hpp"
#include "communication/UpdateGhostParticles.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace communication
{
class GhostLayer
{
private:
    const Subdomain subdomain_;
    IndexView correspondingRealParticle_;
    impl::PeriodicMapping periodicMapping_;
    impl::GhostExchange ghostExchange_;
    impl::UpdateGhostParticles updateGhostParticles_;
    impl::AccumulateForce accumulateForce_;

public:
    void exchangeRealParticles(Particles& particles)
    {
        periodicMapping_.mapIntoDomain(particles);
        Kokkos::fence();
    }

    void createGhostParticles(Particles& particles)
    {
        correspondingRealParticle_ = ghostExchange_.createGhostParticlesXYZ(particles);
        Kokkos::fence();
    }

    void updateGhostParticles(Particles& particles)
    {
        assert(correspondingRealParticle_.extent(0) >= particles.size());

        updateGhostParticles_.updateOnlyPos(particles, correspondingRealParticle_);
        Kokkos::fence();
    }

    void contributeBackGhostToReal(Particles& particles)
    {
        accumulateForce_.ghostToReal(particles, correspondingRealParticle_);
        Kokkos::fence();
    }

    GhostLayer(const Subdomain& subdomain)
        : subdomain_(subdomain),
          periodicMapping_(subdomain),
          ghostExchange_(subdomain),
          updateGhostParticles_(subdomain)
    {
    }
};

}  // namespace communication