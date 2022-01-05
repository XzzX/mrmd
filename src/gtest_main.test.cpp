#include <gtest/gtest.h>
#include <mpi.h>

#include <Kokkos_Core.hpp>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    auto status = RUN_ALL_TESTS();
    if (status != EXIT_SUCCESS) return status;
    MPI_Finalize();
    return EXIT_SUCCESS;
}