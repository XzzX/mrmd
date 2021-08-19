#include "MultiResPeriodicGhostExchange.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
class MultiResPeriodicGhostExchangeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto moleculesPos = molecules.getPos();
        auto moleculesAtomsOffset = molecules.getAtomsOffset();
        auto moleculesNumAtoms = molecules.getNumAtoms();
        int64_t idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    constexpr idx_t moleculeSize = 2;  ///< number of atoms
                    moleculesPos(idx, 0) = x;
                    moleculesPos(idx, 1) = y;
                    moleculesPos(idx, 2) = z;
                    moleculesAtomsOffset(idx) = idx * moleculeSize;
                    moleculesNumAtoms(idx) = moleculeSize;
                    ++idx;
                }
        EXPECT_EQ(idx, 27);
        molecules.numLocalMolecules = 27;
        molecules.numGhostMolecules = 0;
        molecules.resize(molecules.numLocalMolecules + molecules.numGhostMolecules);

        auto atomsPos = atoms.getPos();
        idx = 0;
        for (real_t x = subdomain.minCorner[0] + 0.5_r; x < subdomain.maxCorner[0]; x += 1_r)
            for (real_t y = subdomain.minCorner[1] + 0.5_r; y < subdomain.maxCorner[1]; y += 1_r)
                for (real_t z = subdomain.minCorner[2] + 0.5_r; z < subdomain.maxCorner[2];
                     z += 1_r)
                {
                    atomsPos(idx, 0) = x;
                    atomsPos(idx, 1) = y;
                    atomsPos(idx, 2) = z;
                    ++idx;
                    atomsPos(idx, 0) = x + 0.1_r;
                    atomsPos(idx, 1) = y + 0.2_r;
                    atomsPos(idx, 2) = z + 0.3_r;
                    ++idx;
                }
        EXPECT_EQ(idx, 54);
        atoms.numLocalParticles = 54;
        atoms.numGhostParticles = 0;
        atoms.resize(atoms.numLocalParticles + atoms.numGhostParticles);
    }

    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {3_r, 3_r, 3_r}, 0.7_r);
    data::Molecules molecules = data::Molecules(200);
    data::Particles atoms = data::Particles(200);
};

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
    EXPECT_EQ(this->atoms.numGhostParticles, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(this->subdomain);
    auto correspondingRealParticle = ghostExchange.exchangeGhosts<TypeParam>(
        this->molecules, this->atoms, this->molecules.numLocalMolecules);
    EXPECT_EQ(this->molecules.numGhostMolecules, 9);
    EXPECT_EQ(this->atoms.numGhostParticles, 18);

    auto pos = this->atoms.getPos();
    for (auto idx = this->atoms.numLocalParticles;
         idx < this->atoms.numLocalParticles + this->atoms.numGhostParticles;
         ++idx)
    {
        if constexpr (std::is_same_v<TypeParam, MultiResPeriodicGhostExchange::DIRECTION_X_HIGH>)
        {
            EXPECT_LT(pos(idx, 0), this->subdomain.minCorner[0]);
        }
        if constexpr (std::is_same_v<TypeParam, MultiResPeriodicGhostExchange::DIRECTION_Y_HIGH>)
        {
            EXPECT_LT(pos(idx, 1), this->subdomain.minCorner[1]);
        }
        if constexpr (std::is_same_v<TypeParam, MultiResPeriodicGhostExchange::DIRECTION_Z_HIGH>)
        {
            EXPECT_LT(pos(idx, 2), this->subdomain.minCorner[2]);
        }
        if constexpr (std::is_same_v<TypeParam, MultiResPeriodicGhostExchange::DIRECTION_X_LOW>)
        {
            EXPECT_GT(pos(idx, 0), this->subdomain.minCorner[0]);
        }
        if constexpr (std::is_same_v<TypeParam, MultiResPeriodicGhostExchange::DIRECTION_Y_LOW>)
        {
            EXPECT_GT(pos(idx, 1), this->subdomain.minCorner[1]);
        }
        if constexpr (std::is_same_v<TypeParam, MultiResPeriodicGhostExchange::DIRECTION_Z_LOW>)
        {
            EXPECT_GT(pos(idx, 2), this->subdomain.minCorner[2]);
        }
    }
}

TEST_F(MultiResPeriodicGhostExchangeTest, createGhostParticlesXYZ)
{
    EXPECT_EQ(molecules.numGhostMolecules, 0);
    EXPECT_EQ(atoms.numGhostParticles, 0);
    auto ghostExchange = MultiResPeriodicGhostExchange(subdomain);
    auto correspondingRealParticle = ghostExchange.createGhostParticlesXYZ(molecules, atoms);
    EXPECT_EQ(molecules.numGhostMolecules, 5 * 5 * 5 - 3 * 3 * 3);
    EXPECT_EQ(atoms.numGhostParticles, (5 * 5 * 5 - 3 * 3 * 3) * 2);
}

}  // namespace impl
}  // namespace communication
}  // namespace mrmd