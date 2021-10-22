#pragma once

#include <yaml-cpp/yaml.h>

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
void spartian(YAML::Node& config,
              data::Molecules& molecules,
              data::Atoms& atoms,
              data::Subdomain& subdomain);
}