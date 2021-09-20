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
    analysis::MeanSquareDisplacement msd;
    msd.reset(atoms);
    auto meanSqDisplacement = msd.calc(atoms);
    EXPECT_FLOAT_EQ(meanSqDisplacement, 0_r);
}
}  // namespace analysis
}  // namespace mrmd