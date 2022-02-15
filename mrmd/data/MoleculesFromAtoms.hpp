#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"

namespace mrmd::data
{
data::Molecules createMoleculeForEachAtom(data::Atoms& atoms);
}