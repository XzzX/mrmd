#pragma once

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace io
{
void dumpGRO(const std::string& filename,
             data::Atoms& atoms,
             const data::Subdomain& subdomain,
             const real_t& timestamp,
             const std::string& title,
             bool dumpGhosts = true,
             bool dumpVelocities = false);
}  // namespace io
}  // namespace mrmd