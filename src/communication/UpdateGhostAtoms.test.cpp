#include "communication/UpdateGhostAtoms.hpp"

#include <gtest/gtest.h>

#include <data/Atoms.hpp>

namespace mrmd
{
namespace communication
{
namespace impl
{
struct UpdateGhostAtomsTestData
{
    std::array<real_t, 3> initialDelta;
    std::array<real_t, 3> finalDelta;
};

std::ostream& operator<<(std::ostream& os, const UpdateGhostAtomsTestData& data)
{
    os << "initial delta: [" << data.initialDelta[0] << ", " << data.initialDelta[1] << ", "
       << data.initialDelta[2] << "], ";
    os << "mapped delta: [" << data.finalDelta[0] << ", " << data.finalDelta[1] << ", "
       << data.finalDelta[2] << "]" << std::endl;
    return os;
}

class UpdateGhostAtomsTest : public testing::TestWithParam<UpdateGhostAtomsTestData>
{
protected:
    void SetUp() override
    {
        auto pos = atoms.getPos();
        pos(0, 0) = 0.5_r;
        pos(0, 1) = 0.5_r;
        pos(0, 2) = 0.5_r;
        atoms.numLocalAtoms = 1;
        atoms.numGhostAtoms = 1;

        correspondingRealAtom = IndexView("correspondingRealAtom", 2);
        correspondingRealAtom(0) = -1;
        correspondingRealAtom(1) = 0;
    }

    // void TearDown() override {}

    data::Subdomain subdomain = data::Subdomain({0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, 0.1_r);
    data::Atoms atoms = data::Atoms(2);
    IndexView correspondingRealAtom;
};

TEST_P(UpdateGhostAtomsTest, Check)
{
    auto pos = atoms.getPos();
    pos(1, 0) = 0.5_r + GetParam().initialDelta[0];
    pos(1, 1) = 0.5_r + GetParam().initialDelta[1];
    pos(1, 2) = 0.5_r + GetParam().initialDelta[2];

    UpdateGhostAtoms::updateOnlyPos(atoms, correspondingRealAtom, subdomain);

    EXPECT_FLOAT_EQ(pos(1, 0), 0.5_r + GetParam().finalDelta[0]);
    EXPECT_FLOAT_EQ(pos(1, 1), 0.5_r + GetParam().finalDelta[1]);
    EXPECT_FLOAT_EQ(pos(1, 2), 0.5_r + GetParam().finalDelta[2]);
}

INSTANTIATE_TEST_SUITE_P(
    ShiftX,
    UpdateGhostAtomsTest,
    testing::Values(UpdateGhostAtomsTestData{{0.2_r, 0_r, 0_r}, {1_r, 0_r, 0_r}},
                    UpdateGhostAtomsTestData{{-0.2_r, 0_r, 0_r}, {-1_r, 0_r, 0_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftY,
    UpdateGhostAtomsTest,
    testing::Values(UpdateGhostAtomsTestData{{0_r, 0.2_r, 0_r}, {0_r, 1_r, 0_r}},
                    UpdateGhostAtomsTestData{{0_r, -0.2_r, 0_r}, {0_r, -1_r, 0_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftZ,
    UpdateGhostAtomsTest,
    testing::Values(UpdateGhostAtomsTestData{{0_r, 0_r, 0.2_r}, {0_r, 0_r, 1_r}},
                    UpdateGhostAtomsTestData{{0_r, 0_r, -0.2_r}, {0_r, 0_r, -1_r}}));

INSTANTIATE_TEST_SUITE_P(
    ShiftXYZ,
    UpdateGhostAtomsTest,
    testing::Values(UpdateGhostAtomsTestData{{0.2_r, 0.2_r, 0.2_r}, {1_r, 1_r, 1_r}},
                    UpdateGhostAtomsTestData{{-0.2_r, -0.2_r, -0.2_r}, {-1_r, -1_r, -1_r}}));

}  // namespace impl
}  // namespace communication
}  // namespace mrmd
