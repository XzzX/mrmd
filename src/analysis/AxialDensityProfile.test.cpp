#include "AxialDensityProfile.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
TEST(LinearDensityProfile, histogram)
{
    data::Particles particles(100);
    idx_t idx = 0;
    for (auto i = 0; i < 10; ++i)
    {
        for (auto j = 0; j < i + 1; ++j)
        {
            particles.getPos()(idx, 0) = real_c(i) + 0.5_r;
            ++idx;
        }
    }
    particles.numLocalParticles = idx;

    auto histogram =
        getAxialDensityProfile(particles.getPos(), particles.numLocalParticles, 0_r, 10_r, 10);

    for (auto i = 0; i < 10; ++i)
    {
        EXPECT_FLOAT_EQ(histogram.data(i), real_c(i + 1));
    }
}
}  // namespace analysis
}  // namespace mrmd