#pragma once

#include <Kokkos_Core.hpp>
#include <cassert>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
namespace UpdateGhostAtoms
{
void updateOnlyPos(data::Atoms& atoms,
                   const IndexView& correspondingRealAtom,
                   const data::Subdomain& subdomain);

}  // namespace UpdateGhostAtoms
}  // namespace impl
}  // namespace communication
}  // namespace mrmd