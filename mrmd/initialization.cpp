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
    char const* argv = "";
    // get around warning:
    // ISO C++11 does not allow conversion from string literal to 'char *' [-Wwritable-strings]
    char* argvv = const_cast<char*>(argv);

    initialize(argc, &argvv);
}

void finalize()
{
    Kokkos::finalize();
    MPI_Finalize();
}
}  // namespace mrmd