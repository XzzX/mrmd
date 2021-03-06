#pragma once

#include <yaml-cpp/yaml.h>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
void init(const YAML::Node& config, data::Atoms& atoms, data::Subdomain& subdomain);
data::Molecules initMolecules(const idx_t& numAtoms);
}  // namespace mrmd