#include "VelocityVerlet.hpp"

namespace mrmd
{
namespace action
{
real_t VelocityVerlet::preForceIntegrate(data::Particles& particles, const real_t dt)
{
    auto dtf(0.5_r * dt);
    auto dtv(dt);
    auto pos = particles.getPos();
    auto vel = particles.getVel();
    auto force = particles.getForce();
    auto mass = particles.getMass();

    auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx, real_t& maxDistSqr)
    {
        auto dtfm = dtf / mass(idx);
        vel(idx, 0) += dtfm * force(idx, 0);
        vel(idx, 1) += dtfm * force(idx, 1);
        vel(idx, 2) += dtfm * force(idx, 2);
        auto dx = dtv * vel(idx, 0);
        auto dy = dtv * vel(idx, 1);
        auto dz = dtv * vel(idx, 2);
        pos(idx, 0) += dx;
        pos(idx, 1) += dy;
        pos(idx, 2) += dz;

        auto distSqr = dx * dx + dy * dy + dz * dz;
        if (distSqr > maxDistSqr) maxDistSqr = distSqr;
    };
    real_t maxDistSqr = 0_r;
    Kokkos::parallel_reduce(
        "VelocityVerlet::preForceIntegrate", policy, kernel, Kokkos::Max<real_t>(maxDistSqr));
    Kokkos::fence();
    return std::sqrt(maxDistSqr);
}

void VelocityVerlet::postForceIntegrate(data::Particles& particles, const real_t dt)
{
    auto dtf = 0.5_r * dt;
    auto vel = particles.getVel();
    auto force = particles.getForce();
    auto mass = particles.getMass();

    auto policy = Kokkos::RangePolicy<>(0, particles.numLocalParticles);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        auto dtfm = dtf / mass(idx);
        vel(idx, 0) += dtfm * force(idx, 0);
        vel(idx, 1) += dtfm * force(idx, 1);
        vel(idx, 2) += dtfm * force(idx, 2);
    };
    Kokkos::parallel_for("VelocityVerlet::postForceIntegrate", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd