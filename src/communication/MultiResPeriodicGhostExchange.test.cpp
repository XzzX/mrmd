#include "MultiResPeriodicGhostExchange.hpp"

#include <gtest/gtest.h>

#include "test/GridFixture.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
using MultiResPeriodicGhostExchangeTest = test::GridFixture;

template <typename T>
using TypedMultiResPeriodicGhostExchangeTest = MultiResPeriodicGhostExchangeTest;

using MyTypes = ::testing::Types<MultiResPeriodicGhostExchange::DIRECTION_X_HIGH,
                                 MultiResPeriodicGhostExchange::DIRECTION_X_LOW,
                                 MultiResPeriodicGhostExchange::DIRECTION_Y_HIGH,
                                 MultiResPeriodicGhostExchange::DIRECTION_Y_LOW,
                                 MultiResPeriodicGhostExchange::DIRECTION_Z_HIGH,
                                 MultiResPeriodicGhostExchange::DIRECTION_Z_LOW>;
TYPED_TEST_SUITE(TypedMultiResPeriodicGhostExchangeTest, MyTypes);

TYPED_TEST(TypedMultiResPeriodicGhostExchangeTest, SelfExchange)
{
    EXPECT_EQ(this->molecules.numGhostMolecules, 0);
    EXPECT_EQ(this->atoms.numGhostAtoms, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(this->subdomain);
    auto correspondingRealAtom = ghostExchange.exchangeGhosts<TypeParam>(
        this->molecules, this->atoms, this->molecules.numLocalMolecules);
    EXPECT_EQ(this->molecules.numGhostMolecules, 9);
    EXPECT_EQ(this->atoms.numGhostAtoms, 18);

    auto pos = this->atoms.getPos();
    for (auto idx = this->atoms.numLocalAtoms;
         idx < this->atoms.numLocalAtoms + this->atoms.numGhostAtoms;
         ++idx)
    {
        if constexpr (std::is_same_v<TypeParam,  // NOLINT
                                     MultiResPeriodicGhostExchange::DIRECTION_X_HIGH>)
        {
            EXPECT_LT(pos(idx, 0), this->subdomain.minCorner[0]);
        }
        if constexpr (std::is_same_v<TypeParam,  // NOLINT
                                     MultiResPeriodicGhostExchange::DIRECTION_Y_HIGH>)
        {
            EXPECT_LT(pos(idx, 1), this->subdomain.minCorner[1]);
        }
        if constexpr (std::is_same_v<TypeParam,  // NOLINT
                                     MultiResPeriodicGhostExchange::DIRECTION_Z_HIGH>)
        {
            EXPECT_LT(pos(idx, 2), this->subdomain.minCorner[2]);
        }
        if constexpr (std::is_same_v<TypeParam,  // NOLINT
                                     MultiResPeriodicGhostExchange::DIRECTION_X_LOW>)
        {
            EXPECT_GT(pos(idx, 0), this->subdomain.minCorner[0]);
        }
        if constexpr (std::is_same_v<TypeParam,  // NOLINT
                                     MultiResPeriodicGhostExchange::DIRECTION_Y_LOW>)
        {
            EXPECT_GT(pos(idx, 1), this->subdomain.minCorner[1]);
        }
        if constexpr (std::is_same_v<TypeParam,  // NOLINT
                                     MultiResPeriodicGhostExchange::DIRECTION_Z_LOW>)
        {
            EXPECT_GT(pos(idx, 2), this->subdomain.minCorner[2]);
        }
    }
}

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostAtomsXYZ)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostAtoms, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(subdomain);
    auto correspondingRealAtom = ghostExchange.createGhostAtomsXYZ(molecules, atoms);
    EXPECT_EQ(molecules.numGhostMolecules, 5 * 5 * 5 - 3 * 3 * 3);
    EXPECT_EQ(atoms.numGhostAtoms, (5 * 5 * 5 - 3 * 3 * 3) * 2);
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd