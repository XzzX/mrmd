#include "initialization.hpp"

#include <Kokkos_Core.hpp>

#include "mpi_helper.hpp"

namespace mrmd
{
void initialize(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
}

void initialize()
{
    int argc = 0;
    char* argv = "";

    initialize(argc, &argv);
}

void finalize()
{
    Kokkos::finalize();
    MPI_Finalize();
}
}  // namespace mrmd