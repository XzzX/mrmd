#include "LJ.hpp"
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    LJ();

    return EXIT_SUCCESS;
}