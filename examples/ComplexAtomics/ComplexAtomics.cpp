#include <Kokkos_Core.hpp>

#include "datatypes.hpp"

using namespace mrmd;

struct DoubleCounter
{
    idx_t first = 0;
    idx_t second = 0;

    KOKKOS_INLINE_FUNCTION
    DoubleCounter() = default;
    KOKKOS_INLINE_FUNCTION
    DoubleCounter(idx_t f, idx_t s) : first(f), second(s) {}
    KOKKOS_INLINE_FUNCTION
    DoubleCounter(const DoubleCounter& rhs) = default;

    KOKKOS_INLINE_FUNCTION
    DoubleCounter(const volatile DoubleCounter& rhs)
    {
        first = rhs.first;
        second = rhs.second;
    }

    KOKKOS_INLINE_FUNCTION
    void operator=(const DoubleCounter& rhs) volatile
    {
        first = rhs.first;
        second = rhs.second;
    }
};

DoubleCounter operator+(const DoubleCounter& lhs, const DoubleCounter& rhs)
{
    return DoubleCounter(lhs.first + rhs.first, lhs.second + rhs.second);
}

void complexAtomics()
{
    Kokkos::View<DoubleCounter> hist("a");

    auto policy = Kokkos::RangePolicy<>(0, 1000000);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        DoubleCounter val(idx, 2 * idx);
        auto tmp = Kokkos::atomic_fetch_add(&hist(), val);
        //        hist() = hist() + val;
    };
    Kokkos::parallel_for(policy, kernel);
    Kokkos::fence();

    std::cout << hist().first << " | " << hist().second << std::endl;
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    complexAtomics();

    return EXIT_SUCCESS;
}