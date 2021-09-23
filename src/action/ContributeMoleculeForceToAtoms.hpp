#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class ContributeMoleculeForceToAtoms
{
public:
    static void update(const data::Molecules& molecules, const data::Atoms& atoms);
};
}  // namespace action
}  // namespace mrmd