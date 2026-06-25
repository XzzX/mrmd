// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    void operator+=(const ArrayT& src)
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