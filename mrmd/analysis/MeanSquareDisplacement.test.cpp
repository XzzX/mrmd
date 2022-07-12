#include "MeanSquareDisplacement.hpp"

#include <gtest/gtest.h>

#include "data/Atoms.hpp"
#include "test/SingleAtom.hpp"

namespace mrmd
{
namespace analysis
{
using MeanSquareDisplacementTest = test::SingleAtom;

TEST_F(MeanSquareDisplacementTest, no_displacement)
{
    data::Subdomain subdomain(
        {real_t(0), real_t(0), real_t(0)}, {real_t(10), real_t(10), real_t(10)}, real_t(1));
    analysis::MeanSquareDisplacement msd;
    msd.reset(atoms);
    auto meanSqDisplacement = msd.calc(atoms, subdomain);
    EXPECT_FLOAT_EQ(meanSqDisplacement, real_t(0));
}
}  // namespace analysis
}  // namespace mrmd