#pragma once

#include <Kokkos_Random.hpp>

#include "data/Particles.hpp"
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
