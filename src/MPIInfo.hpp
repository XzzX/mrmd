#pragma once

#include <mpi.h>

namespace mrmd
{
struct MPIInfo
{
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1;
    int size = -1;
};
}  // namespace mrmd