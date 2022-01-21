#pragma once

#include <Kokkos_Core.hpp>
#include <cassert>

#include "data/Atoms.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
namespace AccumulateForce
{
void ghostToReal(data::Atoms& atoms, const IndexView& correspondingRealAtom);
}  // namespace AccumulateForce
}  // namespace impl
}  // namespace communication
}  // namespace mrmd