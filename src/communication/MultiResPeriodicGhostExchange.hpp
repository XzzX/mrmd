#pragma once

#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "util/Kokkos_grow.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class MultiResPeriodicGhostExchange
{
private:
    const data::Subdomain subdomain_;

    data::Particles atoms_ = data::Particles(0);
    data::Particles::pos_t atomPos_;
    Kokkos::View<idx_t> atomNewGhostCounter_;
    Kokkos::View<idx_t>::host_mirror_type hAtomNewGhostCounter_;
    /// Stores the corresponding real particle index for every ghost particle.
    IndexView atomCorrespondingRealParticle_;

    data::Molecules molecules_ = data::Molecules(0);
    data::Molecules::pos_t moleculesPos_;
    data::Molecules::atoms_end_idx_t moleculesAtomEndIdx_;
    Kokkos::View<idx_t> moleculesNewGhostCounter_;
    Kokkos::View<idx_t>::host_mirror_type hMoleculesNewGhostCounter_;
    /// Stores the corresponding real particle index for every ghost particle.
    IndexView moleculesCorrespondingRealParticle_;

public:
    struct DIRECTION_X_HIGH
    {
    };
    struct DIRECTION_X_LOW
    {
    };
    struct DIRECTION_Y_HIGH
    {
    };
    struct DIRECTION_Y_LOW
    {
    };
    struct DIRECTION_Z_HIGH
    {
    };
    struct DIRECTION_Z_LOW
    {
    };

    KOKKOS_INLINE_FUNCTION
    idx_t atomFindRealIdx(const idx_t src) const
    {
        auto realIdx = src;
        while (atomCorrespondingRealParticle_(realIdx) != -1)
        {
            realIdx = atomCorrespondingRealParticle_(realIdx);
            assert(0 <= realIdx);
            assert(realIdx < atoms_.numLocalParticles + atoms_.numGhostParticles);
        }
        return realIdx;
    }

    KOKKOS_INLINE_FUNCTION
    idx_t moleculeFindRealIdx(const idx_t src) const
    {
        auto realIdx = src;
        while (moleculesCorrespondingRealParticle_(realIdx) != -1)
        {
            realIdx = moleculesCorrespondingRealParticle_(realIdx);
            assert(0 <= realIdx);
            assert(realIdx < molecules_.numLocalMolecules + molecules_.numGhostMolecules);
        }
        return realIdx;
    }

    /**
     * @return -1, if no ghost particle was created, idx of new ghost particle otherwise
     */
    KOKKOS_INLINE_FUNCTION
    void copySelfLow(const idx_t idx, const idx_t dim) const
    {
        if (moleculesPos_(idx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto atomsStart = (idx != 0 ? moleculesAtomEndIdx_(idx - 1) : 0);  /// inclusive
            auto atomsEnd = moleculesAtomEndIdx_(idx);                         /// exclusive
            auto moleculeSize = atomsEnd - atomsStart;
            auto moleculeNewGhostIdx = molecules_.numLocalMolecules + molecules_.numGhostMolecules +
                                       Kokkos::atomic_fetch_add(&moleculesNewGhostCounter_(), 1);

            if (moleculeNewGhostIdx < molecules_.size())
            {
                molecules_.copy(moleculeNewGhostIdx, idx);
                moleculesPos_(moleculeNewGhostIdx, dim) += subdomain_.diameter[dim];
                moleculesAtomEndIdx_(moleculeNewGhostIdx) = moleculeNewGhostIdx + 1;
                assert(moleculesPos_(moleculeNewGhostIdx, dim) > subdomain_.maxCorner[dim]);
                assert(moleculesPos_(moleculeNewGhostIdx, dim) < subdomain_.maxGhostCorner[dim]);
                moleculesCorrespondingRealParticle_(moleculeNewGhostIdx) = moleculeFindRealIdx(idx);
            }

            auto atomNewGhostIdx = atoms_.numLocalParticles + atoms_.numGhostParticles +
                                   Kokkos::atomic_fetch_add(&atomNewGhostCounter_(), moleculeSize);

            if (atomNewGhostIdx + moleculeSize - 1 < atoms_.size())
            {
                for (idx_t atomIdx = 0; atomIdx < moleculeSize; ++atomIdx)
                {
                    atoms_.copy(atomNewGhostIdx + atomIdx, atomsStart + atomIdx);
                    atomPos_(atomNewGhostIdx + atomIdx, dim) += subdomain_.diameter[dim];
                    atomCorrespondingRealParticle_(atomNewGhostIdx + atomIdx) =
                        atomFindRealIdx(atomsStart + atomIdx);
                }
            }
        }
    }

    /**
     * @return -1, if no ghost particle was created, idx of new ghost particle otherwise
     */
    KOKKOS_INLINE_FUNCTION
    void copySelfHigh(const idx_t idx, const idx_t dim) const
    {
        if (moleculesPos_(idx, dim) > subdomain_.maxInnerCorner[dim])
        {
            auto atomsStart = (idx != 0 ? moleculesAtomEndIdx_(idx - 1) : 0);  /// inclusive
            auto atomsEnd = moleculesAtomEndIdx_(idx);                         /// exclusive
            auto moleculeSize = atomsEnd - atomsStart;
            auto moleculeNewGhostIdx = molecules_.numLocalMolecules + molecules_.numGhostMolecules +
                                       Kokkos::atomic_fetch_add(&moleculesNewGhostCounter_(), 1);
            if (moleculeNewGhostIdx < molecules_.size())
            {
                molecules_.copy(moleculeNewGhostIdx, idx);
                moleculesPos_(moleculeNewGhostIdx, dim) -= subdomain_.diameter[dim];
                moleculesAtomEndIdx_(moleculeNewGhostIdx) = moleculeNewGhostIdx + 1;
                assert(moleculesPos_(moleculeNewGhostIdx, dim) < subdomain_.minCorner[dim]);
                assert(moleculesPos_(moleculeNewGhostIdx, dim) > subdomain_.minGhostCorner[dim]);
                moleculesCorrespondingRealParticle_(moleculeNewGhostIdx) = moleculeFindRealIdx(idx);
            }

            auto atomNewGhostIdx = atoms_.numLocalParticles + atoms_.numGhostParticles +
                                   Kokkos::atomic_fetch_add(&atomNewGhostCounter_(), moleculeSize);
            if (atomNewGhostIdx + moleculeSize - 1 < atoms_.size())
            {
                for (idx_t atomIdx = 0; atomIdx < moleculeSize; ++atomIdx)
                {
                    atoms_.copy(atomNewGhostIdx + atomIdx, atomsStart + atomIdx);
                    atomPos_(atomNewGhostIdx + atomIdx, dim) -= subdomain_.diameter[dim];
                    atomCorrespondingRealParticle_(atomNewGhostIdx + atomIdx) =
                        atomFindRealIdx(atomsStart + atomIdx);
                }
            }
        }
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_X_HIGH, const idx_t& idx) const { copySelfHigh(idx, 0); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_X_LOW, const idx_t& idx) const { copySelfLow(idx, 0); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_Y_HIGH, const idx_t& idx) const { copySelfHigh(idx, 1); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_Y_LOW, const idx_t& idx) const { copySelfLow(idx, 1); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_Z_HIGH, const idx_t& idx) const { copySelfHigh(idx, 2); }
    KOKKOS_INLINE_FUNCTION
    void operator()(DIRECTION_Z_LOW, const idx_t& idx) const { copySelfLow(idx, 2); }

    template <typename EXCHANGE_DIRECTION>
    IndexView exchangeGhosts(data::Molecules& molecules, data::Particles& atoms, idx_t maxIdx)
    {
        if (atomCorrespondingRealParticle_.extent(0) < atoms.numLocalParticles)
        {
            // initialize atomCorrespondingRealParticle_ for all real particles
            Kokkos::resize(atomCorrespondingRealParticle_, atoms.numLocalParticles);
            Kokkos::deep_copy(atomCorrespondingRealParticle_, -1);
        }
        assert(atomCorrespondingRealParticle_.extent(0) >= atoms.size());

        if (moleculesCorrespondingRealParticle_.extent(0) < molecules.numLocalMolecules)
        {
            // initialize moleculesCorrespondingRealParticle_ for all real particles
            Kokkos::resize(moleculesCorrespondingRealParticle_, molecules.numLocalMolecules);
            Kokkos::deep_copy(moleculesCorrespondingRealParticle_, -1);
        }
        assert(moleculesCorrespondingRealParticle_.extent(0) >= molecules.size());

        auto newMoleculesSize = molecules.numLocalMolecules + molecules.numGhostMolecules;
        auto newAtomsSize = atoms.numLocalParticles + atoms.numGhostParticles;
        do
        {
            if (newMoleculesSize > molecules.size())
            {
                // resize
                molecules.resize(newMoleculesSize);
                util::grow(moleculesCorrespondingRealParticle_, newMoleculesSize);
            }

            if (newAtomsSize > atoms.size())
            {
                // resize
                atoms.resize(newAtomsSize);
                util::grow(atomCorrespondingRealParticle_, newAtomsSize);
            }

            atoms_ = atoms;
            atomPos_ = atoms.getPos();
            Kokkos::deep_copy(atomNewGhostCounter_, 0);

            molecules_ = molecules;
            moleculesPos_ = molecules.getPos();
            moleculesAtomEndIdx_ = molecules.getAtomsEndIdx();
            Kokkos::deep_copy(moleculesNewGhostCounter_, 0);

            for (idx_t idx = 0; idx < maxIdx; ++idx) operator()(EXCHANGE_DIRECTION(), idx);

            Kokkos::deep_copy(hMoleculesNewGhostCounter_, moleculesNewGhostCounter_);
            newMoleculesSize = molecules.numLocalMolecules + molecules.numGhostMolecules +
                               hMoleculesNewGhostCounter_();

            Kokkos::deep_copy(hAtomNewGhostCounter_, atomNewGhostCounter_);
            newAtomsSize =
                atoms.numLocalParticles + atoms.numGhostParticles + hAtomNewGhostCounter_();
        } while ((newMoleculesSize > molecules.size()) ||
                 (newAtomsSize > atoms.size()));  // resize and rerun

        atoms.numGhostParticles += hAtomNewGhostCounter_();
        molecules.numGhostMolecules += hMoleculesNewGhostCounter_();
        return atomCorrespondingRealParticle_;
    }

    IndexView createGhostParticlesXYZ(data::Molecules& molecules, data::Particles& atoms)
    {
        atoms.numGhostParticles = 0;
        util::grow(atomCorrespondingRealParticle_, idx_c(atoms.size()));
        Kokkos::deep_copy(atomCorrespondingRealParticle_, -1);

        molecules.numGhostMolecules = 0;
        util::grow(moleculesCorrespondingRealParticle_, idx_c(molecules.size()));
        Kokkos::deep_copy(moleculesCorrespondingRealParticle_, -1);

        auto maxIdx = molecules.numLocalMolecules + molecules.numGhostMolecules;
        exchangeGhosts<DIRECTION_X_HIGH>(molecules, atoms, maxIdx);
        exchangeGhosts<DIRECTION_X_LOW>(molecules, atoms, maxIdx);
        maxIdx = molecules.numLocalMolecules + molecules.numGhostMolecules;
        exchangeGhosts<DIRECTION_Y_HIGH>(molecules, atoms, maxIdx);
        exchangeGhosts<DIRECTION_Y_LOW>(molecules, atoms, maxIdx);
        maxIdx = molecules.numLocalMolecules + molecules.numGhostMolecules;
        exchangeGhosts<DIRECTION_Z_HIGH>(molecules, atoms, maxIdx);
        exchangeGhosts<DIRECTION_Z_LOW>(molecules, atoms, maxIdx);
        Kokkos::fence();

        molecules.resize(molecules.numLocalMolecules + molecules.numGhostMolecules);
        atoms.resize(atoms.numLocalParticles + atoms.numGhostParticles);
        return atomCorrespondingRealParticle_;
    }

    MultiResPeriodicGhostExchange(const data::Subdomain& subdomain)
        : subdomain_(subdomain),
          atomNewGhostCounter_("atomNewGhostCounter"),
          hAtomNewGhostCounter_("hAtomNewGhostCounter"),
          atomCorrespondingRealParticle_("atomCorrespondingRealParticle", 0),
          moleculesNewGhostCounter_("moleculeNewGhostCounter"),
          hMoleculesNewGhostCounter_("hMoleculeNewGhostCounter"),
          moleculesCorrespondingRealParticle_("moleculeCorrespondingRealParticle", 0)
    {
    }
};
}  // namespace impl
}  // namespace communication
}  // namespace mrmd
