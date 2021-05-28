#include "checks.hpp"
#include "Particles.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>

class KFunc
{
public:
    std::shared_ptr<Particles> particles_ = std::make_shared<Particles>();
    KOKKOS_INLINE_FUNCTION
    void operator() ( const int s, const int a ) const  {
        for (int d = 0; d < Particles::dim; ++d)
            particles_->getPos().access( s, a, d ) += particles_->getVel().access( s, a, d );
    }
};

void functor()
{
    KFunc func;

    Cabana::SimdPolicy<Particles::VectorLength> simd_policy( 0, func.particles_->size() );

    Cabana::simd_parallel_for( simd_policy, func, "vector_op" );
    Kokkos::fence();
}

void aosoa()
{
    Particles particles;

//    for ( std::size_t s = 0; s < particles.numSoA(); ++s )
//        for ( int d = 0; d < particles.dim; ++d )
    int s = 0;
    int d = 0;
    for ( int a = 0; a < 8; ++a )
        particles.getPos().access( s, a, d ) += particles.getVel().access( s, a, d );

}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    aosoa();

    return EXIT_SUCCESS;
}