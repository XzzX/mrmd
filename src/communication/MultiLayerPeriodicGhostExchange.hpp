#pragma once

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
class MultiLayerPeriodicGhostExchange
{
private:
    const data::Subdomain subdomain_;

    data::Particles atoms_ = data::Particles(0);
    data::Particles::pos_t atomPos_;
    Kokkos::View<idx_t> atomNewGhostCounter_;
    Kokkos::View<idx_t>::host_mirror_type hAtomNewGhostCounter_;
    /// Stores the corresponding real particle index for every ghost particle.
    IndexView atomCorrespondingRealParticle_;

    data::Particles molecules_ = data::Particles(0);
    data::Particles::pos_t moleculePos_;
    Kokkos::View<idx_t> moleculeNewGhostCounter_;
    Kokkos::View<idx_t>::host_mirror_type hMoleculeNewGhostCounter_;
    /// Stores the corresponding real particle index for every ghost particle.
    IndexView moleculeCorrespondingRealParticle_;

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
        while (moleculeCorrespondingRealParticle_(realIdx) != -1)
        {
            realIdx = moleculeCorrespondingRealParticle_(realIdx);
            assert(0 <= realIdx);
            assert(realIdx < molecules_.numLocalParticles + molecules_.numGhostParticles);
        }
        return realIdx;
    }

    /**
     * @return -1, if no ghost particle was created, idx of new ghost particle otherwise
     */
    KOKKOS_INLINE_FUNCTION
    void copySelfLow(const idx_t idx, const idx_t dim) const
    {
        if (moleculePos_(idx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto moleculeNewGhostIdx = molecules_.numLocalParticles + molecules_.numGhostParticles +
                                       Kokkos::atomic_fetch_add(&moleculeNewGhostCounter_(), 1);
            if (moleculeNewGhostIdx < molecules_.size())
            {
                molecules_.copy(moleculeNewGhostIdx, idx);
                moleculePos_(moleculeNewGhostIdx, dim) += subdomain_.diameter[dim];
                assert(moleculePos_(moleculeNewGhostIdx, dim) > subdomain_.maxCorner[dim]);
                assert(moleculePos_(moleculeNewGhostIdx, dim) < subdomain_.maxGhostCorner[dim]);
                moleculeCorrespondingRealParticle_(moleculeNewGhostIdx) = moleculeFindRealIdx(idx);
            }

            auto atomNewGhostIdx = atoms_.numLocalParticles + atoms_.numGhostParticles +
                                   Kokkos::atomic_fetch_add(&atomNewGhostCounter_(), 1);
            if (atomNewGhostIdx < atoms_.size())
            {
                atoms_.copy(atomNewGhostIdx, idx);
                atomPos_(atomNewGhostIdx, dim) += subdomain_.diameter[dim];
                atomCorrespondingRealParticle_(atomNewGhostIdx) = atomFindRealIdx(idx);
            }
        }
    }

    /**
     * @return -1, if no ghost particle was created, idx of new ghost particle otherwise
     */
    KOKKOS_INLINE_FUNCTION
    void copySelfHigh(const idx_t idx, const idx_t dim) const
    {
        if (moleculePos_(idx, dim) > subdomain_.maxInnerCorner[dim])
        {
            auto moleculeNewGhostIdx = molecules_.numLocalParticles + molecules_.numGhostParticles +
                                       Kokkos::atomic_fetch_add(&moleculeNewGhostCounter_(), 1);
            if (moleculeNewGhostIdx < molecules_.size())
            {
                molecules_.copy(moleculeNewGhostIdx, idx);
                moleculePos_(moleculeNewGhostIdx, dim) -= subdomain_.diameter[dim];
                assert(moleculePos_(moleculeNewGhostIdx, dim) < subdomain_.minCorner[dim]);
                assert(moleculePos_(moleculeNewGhostIdx, dim) > subdomain_.minGhostCorner[dim]);
                moleculeCorrespondingRealParticle_(moleculeNewGhostIdx) = moleculeFindRealIdx(idx);
            }

            auto atomNewGhostIdx = atoms_.numLocalParticles + atoms_.numGhostParticles +
                                   Kokkos::atomic_fetch_add(&atomNewGhostCounter_(), 1);
            if (atomNewGhostIdx < atoms_.size())
            {
                atoms_.copy(atomNewGhostIdx, idx);
                atomPos_(atomNewGhostIdx, dim) -= subdomain_.diameter[dim];
                atomCorrespondingRealParticle_(atomNewGhostIdx) = atomFindRealIdx(idx);
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
    IndexView exchangeGhosts(data::Particles& molecules, data::Particles& atoms, idx_t maxIdx)
    {
        if (atomCorrespondingRealParticle_.extent(0) < atoms.numLocalParticles)
        {
            // initialize correspondingRealParticle_ for all real particles
            Kokkos::resize(atomCorrespondingRealParticle_, atoms.numLocalParticles);
            Kokkos::deep_copy(atomCorrespondingRealParticle_, -1);
        }
        assert(atomCorrespondingRealParticle_.extent(0) >= atoms.size());

        if (moleculeCorrespondingRealParticle_.extent(0) < molecules.numLocalParticles)
        {
            // initialize correspondingRealParticle_ for all real particles
            Kokkos::resize(moleculeCorrespondingRealParticle_, molecules.numLocalParticles);
            Kokkos::deep_copy(moleculeCorrespondingRealParticle_, -1);
        }
        assert(moleculeCorrespondingRealParticle_.extent(0) >= atoms.size());

        auto newMoleculesSize = molecules.numLocalParticles + molecules.numGhostParticles;
        auto newAtomsSize = atoms.numLocalParticles + atoms.numGhostParticles;
        do
        {
            if (newMoleculesSize > molecules.size())
            {
                // resize
                molecules.resize(newMoleculesSize);
                util::grow(moleculeCorrespondingRealParticle_, newMoleculesSize);
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
            moleculePos_ = molecules.getPos();
            Kokkos::deep_copy(moleculeNewGhostCounter_, 0);

            for (idx_t idx = 0; idx < maxIdx; ++idx) operator()(EXCHANGE_DIRECTION(), idx);

            Kokkos::deep_copy(hMoleculeNewGhostCounter_, moleculeNewGhostCounter_);
            newMoleculesSize = molecules.numLocalParticles + molecules.numGhostParticles +
                               hMoleculeNewGhostCounter_();

            Kokkos::deep_copy(hAtomNewGhostCounter_, atomNewGhostCounter_);
            newAtomsSize =
                atoms.numLocalParticles + atoms.numGhostParticles + hAtomNewGhostCounter_();
        } while ((newMoleculesSize > molecules.size()) ||
                 (newAtomsSize > atoms.size()));  // resize and rerun

        atoms.numGhostParticles += hAtomNewGhostCounter_();
        molecules.numGhostParticles += hMoleculeNewGhostCounter_();
        return atomCorrespondingRealParticle_;
    }

    IndexView createGhostParticlesXYZ(data::Particles& molecules, data::Particles& atoms)
    {
        atoms.numGhostParticles = 0;
        util::grow(atomCorrespondingRealParticle_, idx_c(atoms.size()));
        Kokkos::deep_copy(atomCorrespondingRealParticle_, -1);

        molecules.numGhostParticles = 0;
        util::grow(moleculeCorrespondingRealParticle_, idx_c(molecules.size()));
        Kokkos::deep_copy(moleculeCorrespondingRealParticle_, -1);

        auto maxIdx = molecules.numLocalParticles + molecules.numGhostParticles;
        exchangeGhosts<DIRECTION_X_HIGH>(molecules, atoms, maxIdx);
        exchangeGhosts<DIRECTION_X_LOW>(molecules, atoms, maxIdx);
        maxIdx = molecules.numLocalParticles + molecules.numGhostParticles;
        exchangeGhosts<DIRECTION_Y_HIGH>(molecules, atoms, maxIdx);
        exchangeGhosts<DIRECTION_Y_LOW>(molecules, atoms, maxIdx);
        maxIdx = molecules.numLocalParticles + molecules.numGhostParticles;
        exchangeGhosts<DIRECTION_Z_HIGH>(molecules, atoms, maxIdx);
        exchangeGhosts<DIRECTION_Z_LOW>(molecules, atoms, maxIdx);
        Kokkos::fence();

        molecules.resize(molecules.numLocalParticles + molecules.numGhostParticles);
        atoms.resize(atoms.numLocalParticles + atoms.numGhostParticles);
        return atomCorrespondingRealParticle_;
    }

    MultiLayerPeriodicGhostExchange(const data::Subdomain& subdomain)
        : subdomain_(subdomain),
          atomNewGhostCounter_("atomNewGhostCounter"),
          hAtomNewGhostCounter_("hAtomNewGhostCounter"),
          atomCorrespondingRealParticle_("atomCorrespondingRealParticle", 0),
          moleculeNewGhostCounter_("moleculeNewGhostCounter"),
          hMoleculeNewGhostCounter_("hMoleculeNewGhostCounter"),
          moleculeCorrespondingRealParticle_("moleculeCorrespondingRealParticle", 0)
    {
    }
};
}  // namespace impl
}  // namespace communication
}  // namespace mrmd
