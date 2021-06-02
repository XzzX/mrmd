#include <Kokkos_Core.hpp>

#include "checks.hpp"
#include "datatypes.hpp"

class KokkosVec
{
public:
    constexpr static idx_t size = 10000;
    constexpr static idx_t dim = 3;
    using VectorView = Kokkos::View<real_t**>;

    real_t& get(const idx_t idx, const idx_t dim) const { return vec_(idx, dim); }

private:
    VectorView vec_ = VectorView("vec", size, dim);
};

void kokkos()
{
    KokkosVec vec;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::Serial>(0, KokkosVec::size),
        KOKKOS_LAMBDA(const idx_t idx)
        {
            vec.get(idx, 0) += 5;
            vec.get(idx, 1) += 6;
            vec.get(idx, 2) += 7;
        });
    for (auto i = 0; i < 10; ++i)
    {
        std::cout << vec.get(i, 0) << " | " << vec.get(i, 1) << " | " << vec.get(i, 2) << std::endl;
    }
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    kokkos();

    return EXIT_SUCCESS;
}