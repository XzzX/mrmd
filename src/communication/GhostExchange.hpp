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
    /// Selected indices to be communicated in a certain direction.
    /// first dim: index
    /// second dim: direction
    Kokkos::View<idx_t* [2]> atomsToCommunicateAll_;

    /// number of atoms to communicate in each direction
    Kokkos::View<idx_t[2]> numberOfAtomsToCommunicate_;

    /// Stores the corresponding real atom index for every ghost atom.
    IndexView correspondingRealAtom_;

public:
    IndexView createGhostAtoms(data::Atoms& atoms,
                               const data::Subdomain& subdomain,
                               const idx_t& dim);

    IndexView createGhostAtomsXYZ(data::Atoms& atoms, const data::Subdomain& subdomain);

    void resetCorrespondingRealAtoms(data::Atoms& atoms);

    GhostExchange(const idx_t& initialSize = 100);
};

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
