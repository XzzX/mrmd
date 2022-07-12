#include "VelocityVerletLangevinThermostat.hpp"

#include "util/math.hpp"

namespace mrmd
{
namespace action
{
real_t VelocityVerletLangevinThermostat::preForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    auto RNG = randPool_;
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();
    auto zeta = zeta_;
    auto temperature = temperature_;

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx, real_t& maxDistSqr)
    {
        real_t dx[3];
        dx[0] = pos(idx, 0);
        dx[1] = pos(idx, 1);
        dx[2] = pos(idx, 2);

        auto dtm = dt / mass(idx);
        vel(idx, 0) += real_t(0.5) * dtm * force(idx, 0);
        vel(idx, 1) += real_t(0.5) * dtm * force(idx, 1);
        vel(idx, 2) += real_t(0.5) * dtm * force(idx, 2);

        pos(idx, 0) += real_t(0.5) * dt * vel(idx, 0);
        pos(idx, 1) += real_t(0.5) * dt * vel(idx, 1);
        pos(idx, 2) += real_t(0.5) * dt * vel(idx, 2);

        auto damping = std::exp(-zeta * dtm);
        vel(idx, 0) *= damping;
        vel(idx, 1) *= damping;
        vel(idx, 2) *= damping;

        auto sigma =
            std::sqrt(temperature / mass(idx) * (real_t(1) - std::exp(real_t(-2) * zeta * dtm)));
        // Get a random number state from the pool for the active thread
        auto randGen = RNG.get_state();

        vel(idx, 0) += sigma * randGen.normal();
        vel(idx, 1) += sigma * randGen.normal();
        vel(idx, 2) += sigma * randGen.normal();

        // Give the state back, which will allow another thread to acquire it
        RNG.free_state(randGen);

        pos(idx, 0) += real_t(0.5) * dt * vel(idx, 0);
        pos(idx, 1) += real_t(0.5) * dt * vel(idx, 1);
        pos(idx, 2) += real_t(0.5) * dt * vel(idx, 2);

        dx[0] -= pos(idx, 0);
        dx[1] -= pos(idx, 1);
        dx[2] -= pos(idx, 2);

        auto distSqr = util::dot3(dx, dx);
        if (distSqr > maxDistSqr) maxDistSqr = distSqr;
    };
    real_t maxDistSqr = real_t(0);
    Kokkos::parallel_reduce("VelocityVerletLangevinThermostat::preForceIntegrate",
                            policy,
                            kernel,
                            Kokkos::Max<real_t>(maxDistSqr));
    Kokkos::fence();
    return std::sqrt(maxDistSqr);
}

void VelocityVerletLangevinThermostat::postForceIntegrate(data::Atoms& atoms, const real_t dt)
{
    auto dtf = real_t(0.5) * dt;
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        auto dtfm = dtf / mass(idx);
        vel(idx, 0) += dtfm * force(idx, 0);
        vel(idx, 1) += dtfm * force(idx, 1);
        vel(idx, 2) += dtfm * force(idx, 2);
    };
    Kokkos::parallel_for("VelocityVerletLangevinThermostat::postForceIntegrate", policy, kernel);
    Kokkos::fence();
}

}  // namespace action
}  // namespace mrmd