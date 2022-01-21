#include <Kokkos_Core.hpp>

namespace mrmd::data
{
template <class ScalarType, int N>
struct ArrayT
{
    ScalarType data_[N];

    KOKKOS_INLINE_FUNCTION
    ArrayT() { init(); }

    KOKKOS_INLINE_FUNCTION
    ArrayT(const ArrayT& rhs)
    {
        for (int i = 0; i < N; i++)
        {
            data_[i] = rhs.data_[i];
        }
    }

    KOKKOS_INLINE_FUNCTION
    void init()
    {
        for (int i = 0; i < N; i++)
        {
            data_[i] = 0;
        }
    }

    KOKKOS_INLINE_FUNCTION
    ArrayT& operator+=(const ArrayT& src)
    {
        for (int i = 0; i < N; i++)
        {
            data_[i] += src.data_[i];
        }
        return *this;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile ArrayT& src) volatile
    {
        for (int i = 0; i < N; i++)
        {
            data_[i] += src.data_[i];
        }
    }

    KOKKOS_INLINE_FUNCTION
    ScalarType operator[](idx_t idx) const { return data_[idx]; }
};
}  // namespace mrmd::data

namespace Kokkos
{  // reduction identity must be defined in Kokkos namespace
template <class ScalarType, int N>
struct reduction_identity<mrmd::data::ArrayT<ScalarType, N>>
{
    KOKKOS_FORCEINLINE_FUNCTION static mrmd::data::ArrayT<ScalarType, N> sum()
    {
        return mrmd::data::ArrayT<ScalarType, N>();
    }
};
}  // namespace Kokkos