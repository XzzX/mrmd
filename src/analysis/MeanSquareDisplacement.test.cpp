#include "MeanSquareDisplacement.hpp"

#include <gtest/gtest.h>

#include "data/Particles.hpp"
#include "test/SingleParticle.hpp"

namespace mrmd
{
namespace analysis
{
using MeanSquareDisplacementTest = test::SingleParticle;

TEST_F(MeanSquareDisplacementTest, no_displacement)
{
    data::Subdomain subdomain({0_r, 0_r, 0_r}, {10_r, 10_r, 10_r}, 1_r);
    analysis::MeanSquareDisplacement msd(subdomain);
    msd.reset(atoms);
    auto meanSqDisplacement = msd.calc(atoms);
    EXPECT_FLOAT_EQ(meanSqDisplacement, 0_r);
}
}  // namespace analysis
}  // namespace mrmd