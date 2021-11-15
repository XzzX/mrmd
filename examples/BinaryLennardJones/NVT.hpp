#pragma once

#include <yaml-cpp/yaml.h>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
void nvt(YAML::Node& config, data::Atoms& atoms, const data::Subdomain& subdomain);
}