#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
/**
 * Intended for single process use. Maps atoms that went
 * out of the domain back in a periodic fashion. Mapping is
 * done based on molecule position and atoms are moved according
 * to their molecule.
 *
 * @pre Atom position is at most one periodic copy away
 * from the subdomain.
 * @post Atom position lies within half-open interval
 * [min, max) for all coordinate dimensions.
 */
void realAtomsExchange(const data::Subdomain& subdomain,
                       const data::Molecules& molecules,
                       const data::Atoms& atoms);

}  // namespace communication
}  // namespace mrmd