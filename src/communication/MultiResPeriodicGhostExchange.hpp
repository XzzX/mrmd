#pragma once

#include "data/Molecules.hpp"
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
struct DoubleCounter
{
    idx_t atoms = 0;
    idx_t molecules = 0;

    KOKKOS_INLINE_FUNCTION
    DoubleCounter() = default;
    KOKKOS_INLINE_FUNCTION
    DoubleCounter(idx_t newAtoms, idx_t newMolecules) : atoms(newAtoms), molecules(newMolecules) {}
    KOKKOS_INLINE_FUNCTION
    DoubleCounter(const DoubleCounter& rhs) = default;

    KOKKOS_INLINE_FUNCTION
    DoubleCounter(const volatile DoubleCounter& rhs)
    {
        atoms = rhs.atoms;
        molecules = rhs.molecules;
    }

    KOKKOS_INLINE_FUNCTION
    void operator=(const DoubleCounter& rhs) volatile
    {
        atoms = rhs.atoms;
        molecules = rhs.molecules;
    }

    KOKKOS_INLINE_FUNCTION
    void operator=(volatile const DoubleCounter& rhs) volatile
    {
        atoms = rhs.atoms;
        molecules = rhs.molecules;
    }

    KOKKOS_INLINE_FUNCTION
    DoubleCounter& operator+=(const DoubleCounter& rhs)
    {
        atoms += rhs.atoms;
        molecules += rhs.molecules;
        return *this;
    }
};

KOKKOS_INLINE_FUNCTION
DoubleCounter operator+(const DoubleCounter& lhs, const DoubleCounter& rhs)
{
    return DoubleCounter(lhs.atoms + rhs.atoms, lhs.molecules + rhs.molecules);
}

class MultiResPeriodicGhostExchange
{
private:
    const data::Subdomain subdomain_;

    data::Atoms atoms_ = data::Atoms(0);
    data::Atoms::pos_t atomPos_;
    /// Stores the corresponding real atom index for every ghost atom.
    IndexView atomCorrespondingRealAtom_;

    data::Molecules molecules_ = data::Molecules(0);
    data::Molecules::pos_t moleculesPos_;
    data::Molecules::atoms_offset_t moleculesAtomsOffset_;
    data::Molecules::num_atoms_t moleculesNumAtoms_;
    /// Stores the corresponding real atom index for every ghost atom.
    IndexView moleculesCorrespondingRealAtom_;

    Kokkos::View<idx_t> newMoleculeGhostCounter_;
    Kokkos::View<idx_t>::host_mirror_type hNewMoleculeGhostCounter_;

    Kokkos::View<idx_t> newAtomGhostCounter_;
    Kokkos::View<idx_t>::host_mirror_type hNewAtomGhostCounter_;

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
        while (atomCorrespondingRealAtom_(realIdx) != -1)
        {
            realIdx = atomCorrespondingRealAtom_(realIdx);
            assert(0 <= realIdx);
            assert(realIdx < atoms_.numLocalAtoms + atoms_.numGhostAtoms);
        }
        return realIdx;
    }

    KOKKOS_INLINE_FUNCTION
    idx_t moleculeFindRealIdx(const idx_t src) const
    {
        auto realIdx = src;
        while (moleculesCorrespondingRealAtom_(realIdx) != -1)
        {
            realIdx = moleculesCorrespondingRealAtom_(realIdx);
            assert(0 <= realIdx);
            assert(realIdx < molecules_.numLocalMolecules + molecules_.numGhostMolecules);
        }
        return realIdx;
    }

    /**
     * @return -1, if no ghost atom was created, idx of new ghost atom otherwise
     */
    KOKKOS_INLINE_FUNCTION
    void copySelfLow(const idx_t moleculeIdx, const idx_t dim) const
    {
        if (moleculesPos_(moleculeIdx, dim) < subdomain_.minInnerCorner[dim])
        {
            auto atomsStart = moleculesAtomsOffset_(moleculeIdx);          /// inclusive
            auto atomsEnd = atomsStart + moleculesNumAtoms_(moleculeIdx);  /// exclusive
            auto moleculeSize = atomsEnd - atomsStart;

            auto newMoleculeGhosts = Kokkos::atomic_fetch_add(&newMoleculeGhostCounter_(), 1);
            auto moleculeNewGhostIdx =
                molecules_.numLocalMolecules + molecules_.numGhostMolecules + newMoleculeGhosts;

            auto newAtomGhosts = Kokkos::atomic_fetch_add(&newAtomGhostCounter_(), moleculeSize);
            auto atomNewGhostIdx =
                atoms_.numLocalAtoms + atoms_.numGhostAtoms + newAtomGhosts;

            if (moleculeNewGhostIdx < molecules_.size())
            {
                molecules_.copy(moleculeNewGhostIdx, moleculeIdx);
                moleculesPos_(moleculeNewGhostIdx, dim) += subdomain_.diameter[dim];
                moleculesAtomsOffset_(moleculeNewGhostIdx) = atomNewGhostIdx;
                moleculesNumAtoms_(moleculeNewGhostIdx) = moleculeSize;
                assert(moleculesPos_(moleculeNewGhostIdx, dim) > subdomain_.maxCorner[dim]);
                assert(moleculesPos_(moleculeNewGhostIdx, dim) < subdomain_.maxGhostCorner[dim]);
                moleculesCorrespondingRealAtom_(moleculeNewGhostIdx) =
                    moleculeFindRealIdx(moleculeIdx);
            }

            if (atomNewGhostIdx + moleculeSize - 1 < atoms_.size())
            {
                for (idx_t atomIdx = 0; atomIdx < moleculeSize; ++atomIdx)
                {
                    atoms_.copy(atomNewGhostIdx + atomIdx, atomsStart + atomIdx);
                    atomPos_(atomNewGhostIdx + atomIdx, dim) += subdomain_.diameter[dim];
                    atomCorrespondingRealAtom_(atomNewGhostIdx + atomIdx) =
                        atomFindRealIdx(atomsStart + atomIdx);
                }
            }
        }
    }

    /**
     * @return -1, if no ghost atom was created, moleculeIdx of new ghost atom otherwise
     */
    KOKKOS_INLINE_FUNCTION
    void copySelfHigh(const idx_t moleculeIdx, const idx_t dim) const
    {
        if (moleculesPos_(moleculeIdx, dim) > subdomain_.maxInnerCorner[dim])
        {
            auto atomsStart = moleculesAtomsOffset_(moleculeIdx);          /// inclusive
            auto atomsEnd = atomsStart + moleculesNumAtoms_(moleculeIdx);  /// exclusive
            auto moleculeSize = atomsEnd - atomsStart;

            auto newGhosts = Kokkos::atomic_fetch_add(&newMoleculeGhostCounter_(), 1);
            auto moleculeNewGhostIdx =
                molecules_.numLocalMolecules + molecules_.numGhostMolecules + newGhosts;

            auto newAtomGhosts = Kokkos::atomic_fetch_add(&newAtomGhostCounter_(), moleculeSize);
            auto atomNewGhostIdx =
                atoms_.numLocalAtoms + atoms_.numGhostAtoms + newAtomGhosts;

            if (moleculeNewGhostIdx < molecules_.size())
            {
                molecules_.copy(moleculeNewGhostIdx, moleculeIdx);
                moleculesPos_(moleculeNewGhostIdx, dim) -= subdomain_.diameter[dim];
                moleculesAtomsOffset_(moleculeNewGhostIdx) = atomNewGhostIdx;
                moleculesNumAtoms_(moleculeNewGhostIdx) = moleculeSize;
                assert(moleculesPos_(moleculeNewGhostIdx, dim) < subdomain_.minCorner[dim]);
                assert(moleculesPos_(moleculeNewGhostIdx, dim) > subdomain_.minGhostCorner[dim]);
                moleculesCorrespondingRealAtom_(moleculeNewGhostIdx) =
                    moleculeFindRealIdx(moleculeIdx);
            }

            if (atomNewGhostIdx + moleculeSize - 1 < atoms_.size())
            {
                for (idx_t atomIdx = 0; atomIdx < moleculeSize; ++atomIdx)
                {
                    atoms_.copy(atomNewGhostIdx + atomIdx, atomsStart + atomIdx);
                    atomPos_(atomNewGhostIdx + atomIdx, dim) -= subdomain_.diameter[dim];
                    atomCorrespondingRealAtom_(atomNewGhostIdx + atomIdx) =
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
    IndexView exchangeGhosts(data::Molecules& molecules, data::Atoms& atoms, idx_t maxIdx)
    {
        if (atomCorrespondingRealAtom_.extent(0) < atoms.numLocalAtoms)
        {
            // initialize atomCorrespondingRealAtom_ for all real atoms
            Kokkos::resize(atomCorrespondingRealAtom_, atoms.numLocalAtoms);
            Kokkos::deep_copy(atomCorrespondingRealAtom_, -1);
        }
        assert(atomCorrespondingRealAtom_.extent(0) >= atoms.size());

        if (moleculesCorrespondingRealAtom_.extent(0) < molecules.numLocalMolecules)
        {
            // initialize moleculesCorrespondingRealAtom_ for all real atoms
            Kokkos::resize(moleculesCorrespondingRealAtom_, molecules.numLocalMolecules);
            Kokkos::deep_copy(moleculesCorrespondingRealAtom_, -1);
        }
        assert(moleculesCorrespondingRealAtom_.extent(0) >= molecules.size());

        auto newMoleculesSize = molecules.numLocalMolecules + molecules.numGhostMolecules;
        auto newAtomsSize = atoms.numLocalAtoms + atoms.numGhostAtoms;
        do
        {
            if (newMoleculesSize > molecules.size())
            {
                // resize
                molecules.resize(newMoleculesSize);
                util::grow(moleculesCorrespondingRealAtom_, newMoleculesSize);
            }

            if (newAtomsSize > atoms.size())
            {
                // resize
                atoms.resize(newAtomsSize);
                util::grow(atomCorrespondingRealAtom_, newAtomsSize);
            }

            atoms_ = atoms;
            atomPos_ = atoms.getPos();

            molecules_ = molecules;
            moleculesPos_ = molecules.getPos();
            moleculesAtomsOffset_ = molecules.getAtomsOffset();
            moleculesNumAtoms_ = molecules.getNumAtoms();

            Kokkos::deep_copy(newMoleculeGhostCounter_, 0);
            Kokkos::deep_copy(newAtomGhostCounter_, 0);

            auto policy = Kokkos::RangePolicy<EXCHANGE_DIRECTION>(0, maxIdx);
            Kokkos::parallel_for(policy, *this, "MultiResPeriodicGhostExchange::exchangeGhosts");

            Kokkos::deep_copy(hNewMoleculeGhostCounter_, newMoleculeGhostCounter_);
            newMoleculesSize = molecules.numLocalMolecules + molecules.numGhostMolecules +
                               hNewMoleculeGhostCounter_();

            Kokkos::deep_copy(hNewAtomGhostCounter_, newAtomGhostCounter_);
            newAtomsSize =
                atoms.numLocalAtoms + atoms.numGhostAtoms + hNewAtomGhostCounter_();
        } while ((newMoleculesSize > molecules.size()) ||
                 (newAtomsSize > atoms.size()));  // resize and rerun

        atoms.numGhostAtoms += hNewAtomGhostCounter_();
        molecules.numGhostMolecules += hNewMoleculeGhostCounter_();
        return atomCorrespondingRealAtom_;
    }

    IndexView createGhostAtomsXYZ(data::Molecules& molecules, data::Atoms& atoms)
    {
        // reset ghost atoms
        atoms.numGhostAtoms = 0;
        util::grow(atomCorrespondingRealAtom_, atoms.numLocalAtoms);
        Kokkos::deep_copy(atomCorrespondingRealAtom_, -1);
        atoms.resize(atomCorrespondingRealAtom_.extent(0));

        // reset ghost molecules
        molecules.numGhostMolecules = 0;
        util::grow(moleculesCorrespondingRealAtom_, molecules.numLocalMolecules);
        Kokkos::deep_copy(moleculesCorrespondingRealAtom_, -1);
        molecules.resize(moleculesCorrespondingRealAtom_.extent(0));

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
        atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms);
        return atomCorrespondingRealAtom_;
    }

    MultiResPeriodicGhostExchange(const data::Subdomain& subdomain)
        : subdomain_(subdomain),
          atomCorrespondingRealAtom_("atomCorrespondingRealAtom", 0),
          moleculesCorrespondingRealAtom_("moleculeCorrespondingRealAtom", 0),
          newMoleculeGhostCounter_("newMoleculeGhostCounter_"),
          hNewMoleculeGhostCounter_("hNewMoleculeGhostCounter_"),
          newAtomGhostCounter_("newAtomGhostCounter_"),
          hNewAtomGhostCounter_("hNewAtomGhostCounter_")
    {
    }
};
}  // namespace impl
}  // namespace communication
}  // namespace mrmd
