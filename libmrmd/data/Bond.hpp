#pragma once

#include "datatypes.hpp"

namespace mrmd
{
namespace data
{
struct Bond
{
    idx_t idx;          ///< relative index of first atom
    idx_t jdx;          ///< relative index of second atom
    real_t eqDistance;  ///< eqDistance equilibrium distance of the bond
};

using BondView = Kokkos::View<Bond*>;

}  // namespace data
}  // namespace mrmd