#include "checks.hpp"
#include "Particles.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>

void aosoa()
{
    Particles particles;

    for ( std::size_t s = 0; s < particles.numSoA(); ++s )
        for ( int d = 0; d < particles.dim; ++d )
            for ( std::size_t a = 0; a < particles.arraySize(s); ++a )
                particles.getPos().access( s, a, d ) += particles.getVel().access( s, a, d );
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    aosoa();

    return EXIT_SUCCESS;
}