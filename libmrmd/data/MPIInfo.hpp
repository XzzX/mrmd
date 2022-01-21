#pragma once

#include "mpi_helper.hpp"

namespace mrmd::data
{
struct MPIInfo
{
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = -1;
    int size = -1;

    explicit MPIInfo(MPI_Comm communicator)
    {
        comm = communicator;
        CHECK_MPI(MPI_Comm_rank(communicator, &rank));
        CHECK_MPI(MPI_Comm_size(communicator, &size));
    }
};
}  // namespace mrmd::data