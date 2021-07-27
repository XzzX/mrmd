#include <Kokkos_Core.hpp>

#include "datatypes.hpp"

using namespace mrmd;

void complexAtomics()
{
    Kokkos::View<Kokkos::complex<idx_t>> hist("a");

    auto policy = Kokkos::RangePolicy<>(0, 1000000);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        Kokkos::complex<idx_t> val(idx, 2 * idx);
        atomic_add(&hist(), val);
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    std::cout << hist().real() << " | " << hist().imag() << std::endl;
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    complexAtomics();

    return EXIT_SUCCESS;
}