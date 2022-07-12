#include "PeriodicMapping.hpp"

#include <gtest/gtest.h>

#include "data/Subdomain.hpp"

namespace mrmd
{
namespace communication
{
namespace impl
{
struct TestData
{
    std::array<real_t, 3> initialPos;
    std::array<real_t, 3> mappedPos;
};

std::ostream& operator<<(std::ostream& os, const TestData& data)
{
    os << "initial pos: [" << data.initialPos[0] << ", " << data.initialPos[1] << ", "
       << data.initialPos[2] << "], ";
    os << "mapped pos: [" << data.mappedPos[0] << ", " << data.mappedPos[1] << ", "
       << data.mappedPos[2] << "]" << std::endl;
    return os;
}

class PeriodicMappingTest : public testing::TestWithParam<TestData>
{
};

TEST_P(PeriodicMappingTest, Check)
{
    data::Subdomain subdomain = data::Subdomain(
        {real_t(0), real_t(0), real_t(0)}, {real_t(1), real_t(1), real_t(1)}, real_t(0));
    data::HostAtoms h_atoms(1);
    auto pos = h_atoms.getPos();
    h_atoms.numLocalAtoms = 1;
    pos(0, 0) = GetParam().initialPos[0];
    pos(0, 1) = GetParam().initialPos[1];
    pos(0, 2) = GetParam().initialPos[2];
    data::Atoms atoms(h_atoms);
    PeriodicMapping::mapIntoDomain(atoms, subdomain);
    data::deep_copy(h_atoms, atoms);
    pos = h_atoms.getPos();
    EXPECT_FLOAT_EQ(pos(0, 0), GetParam().mappedPos[0]);
    EXPECT_FLOAT_EQ(pos(0, 1), GetParam().mappedPos[1]);
    EXPECT_FLOAT_EQ(pos(0, 2), GetParam().mappedPos[2]);
}

INSTANTIATE_TEST_SUITE_P(Inside,
                         PeriodicMappingTest,
                         testing::Values(TestData{{real_t(0.4), real_t(0.5), real_t(0.6)},
                                                  {real_t(0.4), real_t(0.5), real_t(0.6)}}));

INSTANTIATE_TEST_SUITE_P(MappingX,
                         PeriodicMappingTest,
                         testing::Values(TestData{{real_t(1.1), real_t(0.5), real_t(0.6)},
                                                  {real_t(0.1), real_t(0.5), real_t(0.6)}},
                                         TestData{{real_t(-0.1), real_t(0.5), real_t(0.6)},
                                                  {real_t(0.9), real_t(0.5), real_t(0.6)}}));

INSTANTIATE_TEST_SUITE_P(MappingY,
                         PeriodicMappingTest,
                         testing::Values(TestData{{real_t(0.4), real_t(1.1), real_t(0.6)},
                                                  {real_t(0.4), real_t(0.1), real_t(0.6)}},
                                         TestData{{real_t(0.4), real_t(-0.1), real_t(0.6)},
                                                  {real_t(0.4), real_t(0.9), real_t(0.6)}}));

INSTANTIATE_TEST_SUITE_P(MappingZ,
                         PeriodicMappingTest,
                         testing::Values(TestData{{real_t(0.4), real_t(0.5), real_t(1.1)},
                                                  {real_t(0.4), real_t(0.5), real_t(0.1)}},
                                         TestData{{real_t(0.4), real_t(0.5), real_t(-0.1)},
                                                  {real_t(0.4), real_t(0.5), real_t(0.9)}}));

INSTANTIATE_TEST_SUITE_P(MappingXYZ,
                         PeriodicMappingTest,
                         testing::Values(TestData{{real_t(1.1), real_t(1.2), real_t(1.3)}, {real_t(0.1), real_t(0.2), real_t(0.3)}},
                                         TestData{{real_t(-0.3), real_t(-0.2), real_t(-0.1)},
                                                  {real_t(0.7), real_t(0.8), real_t(0.9)}}));

}  // namespace impl
}  // namespace communication
}  // namespace mrmd