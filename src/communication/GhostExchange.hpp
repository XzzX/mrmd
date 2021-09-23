#pragma once

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "util/Kokkos_grow.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class GhostExchange
{
private:
    data::Atoms atoms_ = data::Atoms(0);
    const data::Subdomain subdomain_;

    data::Atoms::pos_t pos_;

    Kokkos::View<idx_t> newGhostCounter_;
    Kokkos::View<idx_t>::host_mirror_type hNewGhostCounter_;
    /// Stores the corresponding real atom index for every ghost atom.
    IndexView correspondingRealAtom_;

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
    idx_t findRealIdx(const idx_t src) const
    {
        auto realIdx = src;
        while (correspondingRealAtom_(realIdx) != -1)
        {
            realIdx = correspondingRealAtom_(realIdx);
            assert(0 <= realIdx);
            assert(realIdx < atoms_.numLocalAtoms + atoms_.numGhostAtoms);
        }
        return realIdx;
    }

    /**
     * @return -1, if no ghost atom was created, idx of new ghost atom otherwise
     */
    KOKKOS_INLINE_FUNCTION
    idx_t copySelfLow(const idx_t idx, const idx_t dim) const
    {
        if (pos_(idx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto newGhostIdx = atoms_.numLocalAtoms + atoms_.numGhostAtoms +
                               Kokkos::atomic_fetch_add(&newGhostCounter_(), 1);
            if (newGhostIdx < atoms_.size())
            {
                atoms_.copy(newGhostIdx, idx);
                pos_(newGhostIdx, dim) += subdomain_.diameter[dim];
                assert(pos_(newGhostIdx, dim) >= subdomain_.maxCorner[dim]);
                assert(pos_(newGhostIdx, dim) <= subdomain_.maxGhostCorner[dim]);
                correspondingRealAtom_(newGhostIdx) = findRealIdx(idx);
                return newGhostIdx;
            }
            return -1;
        }
        return -1;
    }

    /**
     * @return -1, if no ghost atom was created, idx of new ghost atom otherwise
     */
    KOKKOS_INLINE_FUNCTION
    idx_t copySelfHigh(const idx_t idx, const idx_t dim) const
    {
        if (pos_(idx, dim) > subdomain_.maxInnerCorner[dim])
        {
            auto newGhostIdx = atoms_.numLocalAtoms + atoms_.numGhostAtoms +
                               Kokkos::atomic_fetch_add(&newGhostCounter_(), 1);
            if (newGhostIdx < atoms_.size())
            {
                atoms_.copy(newGhostIdx, idx);
                pos_(newGhostIdx, dim) -= subdomain_.diameter[dim];
                assert(pos_(newGhostIdx, dim) <= subdomain_.minCorner[dim]);
                assert(pos_(newGhostIdx, dim) >= subdomain_.minGhostCorner[dim]);
                correspondingRealAtom_(newGhostIdx) = findRealIdx(idx);
                return newGhostIdx;
            }
            return -1;
        }
        return -1;
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
    IndexView exchangeGhosts(data::Atoms& atoms, idx_t maxIdx)
    {
        if (correspondingRealAtom_.extent(0) < atoms.numLocalAtoms)
        {
            // initialize correspondingRealAtom_ for all real atoms
            util::grow(correspondingRealAtom_, atoms.numLocalAtoms);
            Kokkos::deep_copy(correspondingRealAtom_, -1);
        }
        assert(correspondingRealAtom_.extent(0) >= atoms.size());

        auto newSize = atoms.numLocalAtoms + atoms.numGhostAtoms;
        do
        {
            if (newSize > atoms.size())
            {
                // resize
                atoms.resize(newSize);
                util::grow(correspondingRealAtom_, newSize);
            }

            atoms_ = atoms;
            pos_ = atoms.getPos();

            Kokkos::deep_copy(newGhostCounter_, 0);

            auto policy = Kokkos::RangePolicy<EXCHANGE_DIRECTION>(0, maxIdx);
            Kokkos::parallel_for(policy, *this, "GhostExchange::exchangeGhosts");
            Kokkos::fence();

            Kokkos::deep_copy(hNewGhostCounter_, newGhostCounter_);
            newSize =
                atoms.numLocalAtoms + atoms.numGhostAtoms + hNewGhostCounter_();
        } while (newSize > atoms.size());  // resize and rerun

        atoms.numGhostAtoms += hNewGhostCounter_();
        return correspondingRealAtom_;
    }

    IndexView createGhostAtomsXYZ(data::Atoms& atoms)
    {
        atoms.numGhostAtoms = 0;
        util::grow(correspondingRealAtom_, atoms.numLocalAtoms);
        Kokkos::deep_copy(correspondingRealAtom_, -1);
        atoms.resize(correspondingRealAtom_.extent(0));

        auto maxIdx = atoms.numLocalAtoms + atoms.numGhostAtoms;
        exchangeGhosts<impl::GhostExchange::DIRECTION_X_HIGH>(atoms, maxIdx);
        exchangeGhosts<impl::GhostExchange::DIRECTION_X_LOW>(atoms, maxIdx);
        maxIdx = atoms.numLocalAtoms + atoms.numGhostAtoms;
        exchangeGhosts<impl::GhostExchange::DIRECTION_Y_HIGH>(atoms, maxIdx);
        exchangeGhosts<impl::GhostExchange::DIRECTION_Y_LOW>(atoms, maxIdx);
        maxIdx = atoms.numLocalAtoms + atoms.numGhostAtoms;
        exchangeGhosts<impl::GhostExchange::DIRECTION_Z_HIGH>(atoms, maxIdx);
        exchangeGhosts<impl::GhostExchange::DIRECTION_Z_LOW>(atoms, maxIdx);
        Kokkos::fence();

        atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms);
        return correspondingRealAtom_;
    }

    GhostExchange(const data::Subdomain& subdomain)
        : subdomain_(subdomain),
          newGhostCounter_("newGhostCounter"),
          hNewGhostCounter_("hNewGhostCounter"),
          correspondingRealAtom_("correspondingRealAtom", 0)
    {
    }
};
}  // namespace impl
}  // namespace communication
}  // namespace mrmd
