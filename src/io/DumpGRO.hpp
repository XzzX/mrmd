#pragma once

#include "data/Particles.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace io
{
void dumpGRO(const std::string& filename,
             data::Particles& particles,
             const data::Subdomain& subdomain,
             const real_t& timestamp,
             const std::string& title,
             bool dumpGhosts = true);
}  // namespace io
}  // namespace mrmd