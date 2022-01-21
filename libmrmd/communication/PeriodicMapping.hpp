#pragma once

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
namespace PeriodicMapping
{
/**
 * @pre Atom position is at most one periodic copy away
 * from the subdomain.
 * @post Atom position lies within half-open interval
 * [min, max) for all coordinate dimensions.
 */
void mapIntoDomain(data::Atoms& atoms, const data::Subdomain& subdomain);
}  // namespace PeriodicMapping
}  // namespace impl
}  // namespace communication
}  // namespace mrmd