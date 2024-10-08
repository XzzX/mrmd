// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "assert/assert.hpp"
#include "data/Atoms.hpp"
#include "data/Bond.hpp"
#include "data/Molecules.hpp"
#include "datatypes.hpp"
#include "util/Kokkos_grow.hpp"
#include "util/math.hpp"

namespace mrmd
{
namespace action
{
namespace impl
{
/**
 * SHAKE algorithm (DOI: 10.1016/0021-9991(77)90098-5)
 * RATTLE algorithm (DOI: 10.1016/0021-9991(83)90014-1)
 */
class Shake
{
private:
    data::Atoms::pos_t pos_;
    data::Atoms::vel_t vel_;
    data::Atoms::force_t::atomic_access_slice force_;
    data::Atoms::mass_t mass_;
    VectorView updatedPos_;

    real_t dtv_;
    real_t dtf_;

public:
    struct UnconstraintUpdate
    {
    };
    struct ApplyConstraint
    {
    };
    struct RemoveBondVelocity
    {
    };

    KOKKOS_INLINE_FUNCTION
    void enforceVelocityConstraint(const idx_t idx, const idx_t jdx, const real_t eqDistance) const
    {
        /// distance vec between atoms
        real_t dist[3];
        dist[0] = pos_(idx, 0) - pos_(jdx, 0);
        dist[1] = pos_(idx, 1) - pos_(jdx, 1);
        dist[2] = pos_(idx, 2) - pos_(jdx, 2);
        /// squared distances between atoms
        auto distSq = util::dot3(dist, dist);

        auto invMassI = 1_r / mass_(idx);
        auto invMassJ = 1_r / mass_(jdx);
        auto reducedMass = 1_r / (invMassI + invMassJ);

        real_t relVel[3];
        relVel[0] = vel_(idx, 0) - vel_(jdx, 0);
        relVel[1] = vel_(idx, 1) - vel_(jdx, 1);
        relVel[2] = vel_(idx, 2) - vel_(jdx, 2);

        auto factor = util::dot3(relVel, dist) / distSq * reducedMass;

        vel_(idx, 0) -= factor * dist[0] * invMassI;
        vel_(idx, 1) -= factor * dist[1] * invMassI;
        vel_(idx, 2) -= factor * dist[2] * invMassI;

        vel_(jdx, 0) += factor * dist[0] * invMassJ;
        vel_(jdx, 1) += factor * dist[1] * invMassJ;
        vel_(jdx, 2) += factor * dist[2] * invMassJ;
    }

    KOKKOS_INLINE_FUNCTION
    void enforcePositionalConstraint(const idx_t idx,
                                     const idx_t jdx,
                                     const real_t eqDistance) const
    {
        /// distance between atoms
        real_t dist[3];
        dist[0] = pos_(idx, 0) - pos_(jdx, 0);
        dist[1] = pos_(idx, 1) - pos_(jdx, 1);
        dist[2] = pos_(idx, 2) - pos_(jdx, 2);
        /// squared distances between atoms
        real_t distSq = util::dot3(dist, dist);

        /// distance between atoms after unconstrained update
        real_t updatedDist[3];
        updatedDist[0] = updatedPos_(idx, 0) - updatedPos_(jdx, 0);
        updatedDist[1] = updatedPos_(idx, 1) - updatedPos_(jdx, 1);
        updatedDist[2] = updatedPos_(idx, 2) - updatedPos_(jdx, 2);
        /// squared distances between updated atoms
        real_t updatedDistSq = util::dot3(updatedDist, updatedDist);

        auto invMassI = 1_r / mass_(idx);
        auto invMassJ = 1_r / mass_(jdx);

        /// coefficient in quadratic equation for lamda, ax**2 + bx + c = 0
        real_t a = util::sqr(invMassI + invMassJ) * distSq;
        real_t b = 2_r * (invMassI + invMassJ) * util::dot3(updatedDist, dist);
        real_t c = updatedDistSq - util::sqr(eqDistance);

        real_t determinant = b * b - 4_r * a * c;
        MRMD_DEVICE_ASSERT_GREATER(determinant, 1e-8);
        determinant = Kokkos::max(0_r, determinant);  // ensure positive determinant

        // solve for lambda
        auto lambda1 = (-b + std::sqrt(determinant)) / (2_r * a);
        auto lambda2 = (-b - std::sqrt(determinant)) / (2_r * a);
        auto lambda = std::abs(lambda1) < std::abs(lambda2) ? lambda1 : lambda2;

        lambda /= dtf_;

        force_(idx, 0) += lambda * dist[0];
        force_(idx, 1) += lambda * dist[1];
        force_(idx, 2) += lambda * dist[2];

        force_(jdx, 0) -= lambda * dist[0];
        force_(jdx, 1) -= lambda * dist[1];
        force_(jdx, 2) -= lambda * dist[2];
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(UnconstraintUpdate, const idx_t idx) const
    {
        auto dtfm = dtf_ / mass_(idx);
        updatedPos_(idx, 0) = pos_(idx, 0) + dtv_ * vel_(idx, 0) + dtfm * force_(idx, 0);
        updatedPos_(idx, 1) = pos_(idx, 1) + dtv_ * vel_(idx, 1) + dtfm * force_(idx, 1);
        updatedPos_(idx, 2) = pos_(idx, 2) + dtv_ * vel_(idx, 2) + dtfm * force_(idx, 2);
    }

    Shake(data::Atoms& atoms, const real_t& dt)
    {
        pos_ = atoms.getPos();
        vel_ = atoms.getVel();
        force_ = atoms.getForce();
        mass_ = atoms.getMass();

        util::grow(updatedPos_, idx_c(atoms.numLocalAtoms + atoms.numGhostAtoms));

        dtv_ = dt;
        dtf_ = 0.5_r * dt * dt;
    }
};
}  // namespace impl

class MoleculeConstraints
{
private:
    idx_t atomsPerMolecule_;
    data::BondView bonds_;

    idx_t numConstraintIterations_;

public:
    void enforcePositionalConstraints(data::Molecules& molecules,
                                      data::Atoms& atoms,
                                      const real_t dt)
    {
        impl::Shake shake(atoms, dt);

        for (int iteration = 0; iteration < numConstraintIterations_; ++iteration)
        {
            auto policy = Kokkos::RangePolicy<impl::Shake::UnconstraintUpdate>(
                0, atoms.numLocalAtoms + atoms.numGhostAtoms);
            Kokkos::parallel_for("Shake::UnconstraintUpdate", policy, shake);
            Kokkos::fence();

            auto moleculesAtomsOffset = molecules.getAtomsOffset();
            auto moleculesNumAtoms = molecules.getNumAtoms();
            auto bonds = bonds_;
            auto applyBondsPolicy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
            auto kernel = KOKKOS_LAMBDA(idx_t moleculeIdx)
            {
                auto atomsStart = moleculesAtomsOffset(moleculeIdx);
                auto numAtoms = moleculesNumAtoms(moleculeIdx);
                for (idx_t bondIdx = 0; bondIdx < bonds.extent(0); ++bondIdx)
                {
                    MRMD_DEVICE_ASSERT_LESS(bonds(bondIdx).idx,
                                            numAtoms,
                                            "not enough atoms in molecule to satisfy bond");
                    MRMD_DEVICE_ASSERT_LESS(bonds(bondIdx).jdx,
                                            numAtoms,
                                            "not enough atoms in molecule to satisfy bond");
                    shake.enforcePositionalConstraint(atomsStart + bonds(bondIdx).idx,
                                                      atomsStart + bonds(bondIdx).jdx,
                                                      bonds(bondIdx).eqDistance);
                }
            };

            Kokkos::parallel_for(
                "MoleculeConstraints::enforcePositionalConstraints", applyBondsPolicy, kernel);
            Kokkos::fence();
        }
    }

    void enforceVelocityConstraints(data::Molecules& molecules, data::Atoms& atoms, const real_t dt)
    {
        impl::Shake shake(atoms, dt);

        auto moleculesAtomsOffset = molecules.getAtomsOffset();
        auto moleculesNumAtoms = molecules.getNumAtoms();
        auto bonds = bonds_;
        auto applyBondsPolicy = Kokkos::RangePolicy<>(0, molecules.numLocalMolecules);
        auto kernel = KOKKOS_LAMBDA(idx_t moleculeIdx)
        {
            auto atomsStart = moleculesAtomsOffset(moleculeIdx);
            auto numAtoms = moleculesNumAtoms(moleculeIdx);
            for (idx_t bondIdx = 0; bondIdx < bonds.extent(0); ++bondIdx)
            {
                MRMD_DEVICE_ASSERT_LESS(
                    bonds(bondIdx).idx, numAtoms, "not enough atoms in molecule to satisfy bond");
                MRMD_DEVICE_ASSERT_LESS(
                    bonds(bondIdx).jdx, numAtoms, "not enough atoms in molecule to satisfy bond");
                shake.enforceVelocityConstraint(atomsStart + bonds(bondIdx).idx,
                                                atomsStart + bonds(bondIdx).jdx,
                                                bonds(bondIdx).eqDistance);
            }
        };
        Kokkos::parallel_for(
            "MoleculeConstraints::enforceVelocityConstraints", applyBondsPolicy, kernel);
        Kokkos::fence();
    }

    void setConstraints(const data::BondView& bonds) { bonds_ = bonds; }
    void setConstraints(const data::BondView::host_mirror_type& bonds)
    {
        Kokkos::resize(bonds_, bonds.extent(0));
        Kokkos::deep_copy(bonds_, bonds);
    }

    MoleculeConstraints(idx_t atomsPerMolecule, idx_t numConstraintIterations)
        : atomsPerMolecule_(atomsPerMolecule), numConstraintIterations_(numConstraintIterations)
    {
    }
};

}  // namespace action
}  // namespace mrmd