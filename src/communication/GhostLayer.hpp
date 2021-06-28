#pragma once

#include "communication/AccumulateForce.hpp"
#include "communication/GhostExchange.hpp"
#include "communication/PeriodicMapping.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace communication
{
class GhostLayer
{
private:
    const Subdomain subdomain_;

public:
    void exchangeRealParticles(Particles& particles)
    {
        impl::PeriodicMapping mapping(subdomain_);
        mapping.mapIntoDomain(particles);
        Kokkos::fence();
    }

    void createGhostParticles(Particles& particles)
    {
        impl::GhostExchange ghostExchange(subdomain_);
        ghostExchange.createGhostParticlesXYZ(particles);
        Kokkos::fence();
    }

    void updateGhostParticles(Particles& particles) {}

    void contributeBackGhostToReal(Particles& particles)
    {
        impl::AccumulateForce accumulate;
        accumulate.ghostToReal(particles);
        Kokkos::fence();
    }

    GhostLayer(const Subdomain& subdomain) : subdomain_(subdomain) {}
};

}  // namespace communication