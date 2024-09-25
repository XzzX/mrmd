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

#pragma once

#include <Kokkos_Random.hpp>

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace util
{
struct Random
{
    // The GeneratorPool
    Kokkos::Random_XorShift1024_Pool<> randPool_;

    Random() : randPool_(1234) {}

    KOKKOS_INLINE_FUNCTION
    real_t draw() const
    {
        // Get a random number state from the pool for the active thread
        auto randGen = randPool_.get_state();

        auto tmp = randGen.drand();

        // Give the state back, which will allow another thread to acquire it
        randPool_.free_state(randGen);

        return tmp;
    }
};
}  // namespace util
}  // namespace mrmd
