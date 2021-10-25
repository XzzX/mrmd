#include "AxialDensityProfile.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
/**
 * increasing density of type 0 atoms from left to right
 * decreasing density of type 1 atoms from right to left
 */
TEST(LinearDensityProfile, histogram)
{
    data::Atoms atoms(100 * 2);
    atoms.numLocalAtoms = 200;

    auto policy = Kokkos::RangePolicy<>(0, 1);
    auto kernel = KOKKOS_LAMBDA(const idx_t& /*tmp*/, idx_t& sum)
    {
        idx_t idx = 0;
        for (auto i = 0; i < 10; ++i)
        {
            for (auto j = 0; j < i + 1; ++j)
            {
                atoms.getPos()(idx, 0) = real_c(i) + 0.5_r;
                atoms.getType()(idx) = 0;
                ++idx;

                atoms.getPos()(idx, 0) = 10_r - (real_c(i) + 0.5_r);
                atoms.getType()(idx) = 1;
                ++idx;
            }
        }
        sum += idx;
    };
    idx_t numAtoms = 0;
    Kokkos::parallel_reduce("LinearDensityProfile::histogram", policy, kernel, numAtoms);
    Kokkos::fence();

    auto histogram =
        getAxialDensityProfile(numAtoms, atoms.getPos(), atoms.getType(), 2, 0_r, 10_r, 10);

    for (auto i = 0; i < 10; ++i)
    {
        EXPECT_FLOAT_EQ(histogram[0].data(i), real_c(i + 1));
        EXPECT_FLOAT_EQ(histogram[1].data(i), real_c(11 - (i + 1)));
    }
}
}  // namespace analysis
}  // namespace mrmd