#pragma once

#include <mpi.h>

#include "assert.hpp"

namespace mrmd
{
inline void CHECK_MPI(const int& status) { CHECK_EQUAL(status, MPI_SUCCESS); }
}  // namespace mrmd