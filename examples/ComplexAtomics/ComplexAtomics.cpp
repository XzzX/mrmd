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
#include <iostream>

#include "datatypes.hpp"

using namespace mrmd;

struct DoubleCounter
{
    idx_t first = 0;
    idx_t second = 0;

    KOKKOS_INLINE_FUNCTION
    DoubleCounter() = default;
    KOKKOS_INLINE_FUNCTION
    DoubleCounter(idx_t firstArg, idx_t secondArg) : first(firstArg), second(secondArg) {}
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

    KOKKOS_INLINE_FUNCTION
    void operator=(volatile const DoubleCounter& rhs) volatile
    {
        first = rhs.first;
        second = rhs.second;
    }

    KOKKOS_INLINE_FUNCTION
    DoubleCounter& operator+=(const DoubleCounter& rhs)
    {
        first += rhs.first;
        second += rhs.second;
        return *this;
    }
};

KOKKOS_INLINE_FUNCTION
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
