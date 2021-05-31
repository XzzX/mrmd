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

    auto pos = particles.getPos();
    auto vel = particles.getVel();
    for ( std::size_t s = 0; s < particles.numSoA(); ++s )
        for ( int d = 0; d < particles.dim; ++d )
            for ( int a = 0; a < 32; ++a )
                pos.access( s, a, d ) += vel.access( s, a, d );
    std::cout << pos.access( 0, 0, 0 ) << std::endl;
}

void cabana_simd()
{
    Particles particles;

    auto pos = particles.getPos();
    auto vel = particles.getVel();

    auto vector_kernel =
        KOKKOS_LAMBDA( const int s, const int a )
        {
          pos.access(s,a, 0) = vel.access(s,a, 0);
        };

    using ExecutionSpace = Kokkos::Serial;
    Cabana::SimdPolicy<8,ExecutionSpace> simd_policy( 0, 100 );

    Cabana::simd_parallel_for( simd_policy, vector_kernel, "vector_op" );

    std::cout << pos.access( 0, 0, 0 ) << std::endl;
}

void kokkos()
{
    using VecView = Kokkos::View<double**, Kokkos::LayoutRight>;

    VecView pos = VecView("pos", 3, 100);
    VecView vel = VecView("pos", 3, 100);

    for ( int d = 0; d < 3; ++d )
        for ( int a = 0; a < 100; ++a )
            pos(d, a) += vel( d, a );
    std::cout << &pos( 0, 0 ) << std::endl;
    std::cout << &pos( 0, 1 ) << std::endl;
}

void kokkos_for()
{
    using VecView = Kokkos::View<double**, Kokkos::LayoutRight>;

    VecView pos = VecView("pos", 3, 100);
    VecView vel = VecView("pos", 3, 100);

    Kokkos::parallel_for("loop",
                         100,
                         KOKKOS_LAMBDA(const int idx)
                         {
                           pos(0, idx) += vel( 0, idx );
                           pos(1, idx) += vel( 1, idx );
                           pos(2, idx) += vel( 2, idx );
                         });

    std::cout << pos( 0, 0 ) << std::endl;
    std::cout << pos( 0, 1 ) << std::endl;
}