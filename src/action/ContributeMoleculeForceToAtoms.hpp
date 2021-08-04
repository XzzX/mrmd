#pragma once

#include "data/Molecules.hpp"
#include "data/Particles.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
class ContributeMoleculeForceToAtoms
{
public:
    static void update(const data::Molecules& molecules, const data::Particles& atoms);
};
}  // namespace action
}  // namespace mrmd