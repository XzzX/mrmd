#include "GhostExchange.hpp"

#include <fmt/format.h>

#include <assert.hpp>

namespace mrmd
{
namespace communication
{
namespace impl
{
IndexView GhostExchange::createGhostAtoms(data::Atoms& atoms,
                                          const data::Subdomain& subdomain,
                                          const idx_t& dim)
{
    ASSERT_LESSEQUAL(0, dim);
    ASSERT_LESS(dim, 3);

    auto pos = atoms.getPos();

    auto h_numberOfAtomsToCommunicate =
        Kokkos::create_mirror_view(Kokkos::HostSpace(), numberOfAtomsToCommunicate_);

    auto newAtoms = 0;
    do
    {
        // reset selected atoms
        util::grow(atomsToCommunicateAll_, newAtoms);
        Kokkos::deep_copy(atomsToCommunicateAll_, -1);
        Kokkos::deep_copy(numberOfAtomsToCommunicate_, 0);

        auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms + atoms.numGhostAtoms);
        auto kernel = KOKKOS_CLASS_LAMBDA(idx_t idx)
        {
            if (pos(idx, dim) < subdomain.minInnerCorner[dim])
            {
                auto commIdx = Kokkos::atomic_fetch_add(&numberOfAtomsToCommunicate_(0), 1);
                if (commIdx < idx_c(atomsToCommunicateAll_.extent(0)))
                {
                    atomsToCommunicateAll_(commIdx, 0) = idx;
                }
            }

            if (pos(idx, dim) >= subdomain.maxInnerCorner[dim])
            {
                auto commIdx = Kokkos::atomic_fetch_add(&numberOfAtomsToCommunicate_(1), 1);
                if (commIdx < idx_c(atomsToCommunicateAll_.extent(0)))
                {
                    atomsToCommunicateAll_(commIdx, 1) = idx;
                }
            }
        };
        Kokkos::parallel_for(fmt::format("GhostExchange::selectAtoms_{}", dim), policy, kernel);
        Kokkos::fence();
        Kokkos::deep_copy(h_numberOfAtomsToCommunicate, numberOfAtomsToCommunicate_);
        newAtoms = std::max(h_numberOfAtomsToCommunicate(0), h_numberOfAtomsToCommunicate(1));
    } while (newAtoms > idx_c(atomsToCommunicateAll_.extent(0)));

    atoms.resize(atoms.numLocalAtoms + atoms.numGhostAtoms + h_numberOfAtomsToCommunicate(0) +
                 h_numberOfAtomsToCommunicate(1));
    pos = atoms.getPos();

    util::grow(correspondingRealAtom_,
               atoms.numLocalAtoms + atoms.numGhostAtoms + h_numberOfAtomsToCommunicate(0) +
                   h_numberOfAtomsToCommunicate(1));

    {
        auto policy = Kokkos::RangePolicy<>(0, newAtoms);
        auto kernel = KOKKOS_CLASS_LAMBDA(const idx_t idx)
        {
            if (idx < numberOfAtomsToCommunicate_(0))
            {
                auto newGhostIdx = atoms.numLocalAtoms + atoms.numGhostAtoms + idx;
                atoms.copy(newGhostIdx, atomsToCommunicateAll_(idx, 0));
                pos(newGhostIdx, dim) += subdomain.diameter[dim];
                ASSERT_GREATEREQUAL(pos(newGhostIdx, dim), subdomain.maxCorner[dim]);
                ASSERT_LESSEQUAL(pos(newGhostIdx, dim), subdomain.maxGhostCorner[dim]);
                auto realIdx = atomsToCommunicateAll_(idx, 0);
                while (correspondingRealAtom_(realIdx) != -1)
                {
                    realIdx = correspondingRealAtom_(realIdx);
                    ASSERT_LESSEQUAL(0, realIdx);
                    ASSERT_LESS(realIdx, atoms.numLocalAtoms + atoms.numGhostAtoms);
                }
                correspondingRealAtom_(newGhostIdx) = realIdx;
            }

            if (idx < numberOfAtomsToCommunicate_(1))
            {
                auto newGhostIdx = atoms.numLocalAtoms + atoms.numGhostAtoms +
                                   numberOfAtomsToCommunicate_(0) + idx;
                atoms.copy(newGhostIdx, atomsToCommunicateAll_(idx, 1));
                pos(newGhostIdx, dim) -= subdomain.diameter[dim];
                ASSERT_LESSEQUAL(pos(newGhostIdx, dim), subdomain.minCorner[dim]);
                ASSERT_GREATEREQUAL(pos(newGhostIdx, dim), subdomain.minGhostCorner[dim]);
                auto realIdx = atomsToCommunicateAll_(idx, 1);
                while (correspondingRealAtom_(realIdx) != -1)
                {
                    realIdx = correspondingRealAtom_(realIdx);
                    ASSERT_LESSEQUAL(0, realIdx);
                    ASSERT_LESS(realIdx, atoms.numLocalAtoms + atoms.numGhostAtoms);
                }
                correspondingRealAtom_(newGhostIdx) = realIdx;
            }
        };
        Kokkos::parallel_for(fmt::format("GhostExchange::copyAtoms_{}", dim), policy, kernel);
        Kokkos::fence();
    }
    atoms.numGhostAtoms += h_numberOfAtomsToCommunicate(0) + h_numberOfAtomsToCommunicate(1);

    return correspondingRealAtom_;
}

IndexView GhostExchange::createGhostAtomsXYZ(data::Atoms& atoms, const data::Subdomain& subdomain)
{
    resetCorrespondingRealAtoms(atoms);
    atoms.numGhostAtoms = 0;

    createGhostAtoms(atoms, subdomain, COORD_X);
    createGhostAtoms(atoms, subdomain, COORD_Y);
    createGhostAtoms(atoms, subdomain, COORD_Z);

    return correspondingRealAtom_;
}

void GhostExchange::resetCorrespondingRealAtoms(data::Atoms& atoms)
{
    util::grow(correspondingRealAtom_, atoms.numLocalAtoms);
    Kokkos::deep_copy(correspondingRealAtom_, -1);
}

GhostExchange::GhostExchange(const idx_t& initialSize)
    : atomsToCommunicateAll_("atomsToCommunicateAll", initialSize),
      numberOfAtomsToCommunicate_("numberOfAtomsToCommunicate"),
      correspondingRealAtom_("correspondingRealAtom", initialSize)
{
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
