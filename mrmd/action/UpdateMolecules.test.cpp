#include "UpdateMolecules.hpp"

#include <gtest/gtest.h>

#include "test/DiamondFixture.hpp"

namespace mrmd
{
namespace action
{
using UpdateMoleculesTest = test::DiamondFixture;

struct Weight
{
    KOKKOS_INLINE_FUNCTION
    void operator()(const real_t x,
                    const real_t y,
                    const real_t z,
                    real_t& lambda,
                    real_t& modulatedLambda,
                    real_t& gradLambdaX,
                    real_t& gradLambdaY,
                    real_t& gradLambdaZ) const
    {
        modulatedLambda = x > 0 ? real_t(0.7) : real_t(-0.7);
    }
};

TEST_F(UpdateMoleculesTest, update)
{
    Weight weight;
    action::UpdateMolecules::update(molecules, atoms, weight);

    data::HostMolecules h_molecules(molecules);
    EXPECT_FLOAT_EQ(h_molecules.getPos()(0, 0), real_t(0.25));
    EXPECT_FLOAT_EQ(h_molecules.getPos()(0, 1), real_t(0.75));
    EXPECT_FLOAT_EQ(h_molecules.getPos()(0, 2), real_t(0));

    EXPECT_FLOAT_EQ(h_molecules.getModulatedLambda()(0), real_t(0.7));

    EXPECT_FLOAT_EQ(h_molecules.getPos()(1, 0), real_t(-0.25));
    EXPECT_FLOAT_EQ(h_molecules.getPos()(1, 1), real_t(-0.75));
    EXPECT_FLOAT_EQ(h_molecules.getPos()(1, 2), real_t(0));

    EXPECT_FLOAT_EQ(h_molecules.getModulatedLambda()(1), real_t(-0.7));
}

}  // namespace action
}  // namespace mrmd