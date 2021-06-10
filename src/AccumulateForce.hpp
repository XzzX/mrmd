#pragma once

#include "Particles.hpp"
#include "Subdomain.hpp"

class AccumulateForce
{
private:
    Particles& particles_;

public:
    KOKKOS_INLINE_FUNCTION
    void operator()(const idx_t& idx) const
    {
        if (particles_.getGhost()(idx) == -1) return;

        auto realIdx = particles_.getGhost()(idx);
        ASSERT_EQUAL(
            particles_.getGhost()(realIdx), -1, "We do not want to add forces to ghost particles!");
        for (auto dim = 0; dim < Particles::dim; ++dim)
        {
            particles_.getForce()(realIdx, dim) += particles_.getForce()(idx, dim);
            particles_.getForce()(idx, dim) = 0_r;
        }
    }

    AccumulateForce(Particles& particles) : particles_(particles) {}
};
