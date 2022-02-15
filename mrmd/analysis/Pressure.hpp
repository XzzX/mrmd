#pragma once

#include "../data/Subdomain.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd
{
namespace analysis
{
real_t getPressure(data::Atoms& atoms, const data::Subdomain& subdomain);

}  // namespace analysis
}  // namespace mrmd