#pragma once

#include "data/Particles.hpp"
#include "datatypes.hpp"
#include "util/Kokkos_grow.hpp"

namespace mrmd
{
namespace action
{
namespace impl
{
/**
 * SHAKE algorithm (DOI: 10.1016/0021-9991(77)90098-5)
 */
class Shake
{
private:
    data::Particles::pos_t pos_;
    data::Particles::vel_t vel_;
    data::Particles::force_t::atomic_access_slice force_;
    VectorView updatedPos_;

    PairView bondedAtoms_;
    ScalarView bondEquilibriumLength_;

    real_t dtv_;
    real_t dtfsq_;

public:
    struct UnconstraintUpdate
    {
    };
    struct ApplyConstraint
    {
    };

    auto getUpdatedPos() const { return updatedPos_; }

    void setBonds(const PairView::host_mirror_type& bondedAtoms,
                  const ScalarView::host_mirror_type& bondEquilibriumLength)
    {
        assert(bondedAtoms.extent(0) == bondEquilibriumLength.extent(0));
        Kokkos::resize(bondedAtoms_, bondedAtoms.extent(0));
        Kokkos::deep_copy(bondedAtoms_, bondedAtoms);
        Kokkos::resize(bondEquilibriumLength_, bondEquilibriumLength.extent(0));
        Kokkos::deep_copy(bondEquilibriumLength_, bondEquilibriumLength);
    }

    KOKKOS_INLINE_FUNCTION void applyConstraint(const idx_t idx,
                                                const idx_t jdx,
                                                const real_t eqDistance) const
    {
        /// distance vec between atoms, with PBC
        double r01[3];
        r01[0] = pos_(idx, 0) - pos_(jdx, 0);
        r01[1] = pos_(idx, 1) - pos_(jdx, 1);
        r01[2] = pos_(idx, 2) - pos_(jdx, 2);
        //        domain->minimum_image(r01);

        /// distance vec after unconstrained update, with PBC
        double s01[3];
        s01[0] = updatedPos_(idx, 0) - updatedPos_(jdx, 0);
        s01[1] = updatedPos_(idx, 1) - updatedPos_(jdx, 1);
        s01[2] = updatedPos_(idx, 2) - updatedPos_(jdx, 2);
        //        domain->minimum_image_once(s01);

        /// squared distances between particles
        double r01sq = r01[0] * r01[0] + r01[1] * r01[1] + r01[2] * r01[2];
        /// squared distances between updated particles
        double s01sq = s01[0] * s01[0] + s01[1] * s01[1] + s01[2] * s01[2];

        auto invMassI = 1_r;
        auto invMassJ = 1_r;

        /// coefficient in quadratic equation for lamda, ax**2 + bx + c = 0
        double a = (invMassI + invMassJ) * (invMassI + invMassJ) * r01sq;
        double b =
            2_r * (invMassI + invMassJ) * (s01[0] * r01[0] + s01[1] * r01[1] + s01[2] * r01[2]);
        double c = s01sq - eqDistance * eqDistance;

        double determinant = b * b - 4_r * a * c;
        assert(determinant >= 0);

        // solve for lambda
        auto lambda1 = (-b + std::sqrt(determinant)) / (2_r * a);
        auto lambda2 = (-b - std::sqrt(determinant)) / (2_r * a);
        auto lambda = std::fabs(lambda1) < std::fabs(lambda2) ? lambda1 : lambda2;

        lambda /= dtfsq_;

        force_(idx, 0) += lambda * r01[0];
        force_(idx, 1) += lambda * r01[1];
        force_(idx, 2) += lambda * r01[2];

        force_(jdx, 0) -= lambda * r01[0];
        force_(jdx, 1) -= lambda * r01[1];
        force_(jdx, 2) -= lambda * r01[2];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(ApplyConstraint, const idx_t idx) const
    {
        applyConstraint(bondedAtoms_(idx, 0), bondedAtoms_(idx, 1), bondEquilibriumLength_(idx));
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(UnconstraintUpdate, const idx_t idx) const
    {
        auto dtfmsq = dtfsq_;
        updatedPos_(idx, 0) = pos_(idx, 0) + dtv_ * vel_(idx, 0) + dtfmsq * force_(idx, 0);
        updatedPos_(idx, 1) = pos_(idx, 1) + dtv_ * vel_(idx, 1) + dtfmsq * force_(idx, 1);
        updatedPos_(idx, 2) = pos_(idx, 2) + dtv_ * vel_(idx, 2) + dtfmsq * force_(idx, 2);
    }

    void apply(data::Particles& particles)
    {
        pos_ = particles.getPos();
        vel_ = particles.getVel();
        force_ = particles.getForce();

        util::grow(updatedPos_, pos_.extent(0));

        auto policy = Kokkos::RangePolicy<UnconstraintUpdate>(
            0, particles.numLocalParticles + particles.numGhostParticles);
        Kokkos::parallel_for("Shake::UnconstraintUpdate", policy, *this);
        Kokkos::fence();

        auto applyConstraintPolicy =
            Kokkos::RangePolicy<ApplyConstraint>(0, bondedAtoms_.extent(0));
        Kokkos::parallel_for("Shake::ApplyConstraint", applyConstraintPolicy, *this);
        Kokkos::fence();
    }

    Shake(const real_t& dt)
    {
        dtv_ = dt;
        dtfsq_ = 0.5_r * dt * dt;
    }
};
}  // namespace impl

}  // namespace action
}  // namespace mrmd