#pragma once

#include <yaml-cpp/yaml.h>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
void npt(YAML::Node& config, data::Atoms& atoms, data::Subdomain& subdomain);
}