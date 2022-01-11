#pragma once

#include <mpi.h>

#include "assert/assert.hpp"

namespace mrmd
{
inline void CHECK_MPI(const int& status) { MRMD_HOST_CHECK_EQUAL(status, MPI_SUCCESS); }
}  // namespace mrmd