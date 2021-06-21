#include "PeriodicMapping.hpp"

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "data/Subdomain.hpp"

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
    Subdomain subdomain = Subdomain({0_r, 0_r, 0_r}, {1_r, 1_r, 1_r}, 0_r);
    Particles particles;
    particles.numLocalParticles = 1;
    particles.getPos(0, 0) = GetParam().initialPos[0];
    particles.getPos(0, 1) = GetParam().initialPos[1];
    particles.getPos(0, 2) = GetParam().initialPos[2];
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Serial>(0, particles.numLocalParticles),
                         PeriodicMapping(particles, subdomain));
    EXPECT_FLOAT_EQ(particles.getPos(0, 0), GetParam().mappedPos[0]);
    EXPECT_FLOAT_EQ(particles.getPos(0, 1), GetParam().mappedPos[1]);
    EXPECT_FLOAT_EQ(particles.getPos(0, 2), GetParam().mappedPos[2]);
}

INSTANTIATE_TEST_SUITE_P(Inside,
                         PeriodicMappingTest,
                         testing::Values(TestData{{0.4_r, 0.5_r, 0.6_r}, {0.4_r, 0.5_r, 0.6_r}}));

INSTANTIATE_TEST_SUITE_P(MappingX,
                         PeriodicMappingTest,
                         testing::Values(TestData{{1.1_r, 0.5_r, 0.6_r}, {0.1_r, 0.5_r, 0.6_r}},
                                         TestData{{-0.1_r, 0.5_r, 0.6_r}, {0.9_r, 0.5_r, 0.6_r}}));

INSTANTIATE_TEST_SUITE_P(MappingY,
                         PeriodicMappingTest,
                         testing::Values(TestData{{0.4_r, 1.1_r, 0.6_r}, {0.4_r, 0.1_r, 0.6_r}},
                                         TestData{{0.4_r, -0.1_r, 0.6_r}, {0.4_r, 0.9_r, 0.6_r}}));

INSTANTIATE_TEST_SUITE_P(MappingZ,
                         PeriodicMappingTest,
                         testing::Values(TestData{{0.4_r, 0.5_r, 1.1_r}, {0.4_r, 0.5_r, 0.1_r}},
                                         TestData{{0.4_r, 0.5_r, -0.1_r}, {0.4_r, 0.5_r, 0.9_r}}));

INSTANTIATE_TEST_SUITE_P(MappingXYZ,
                         PeriodicMappingTest,
                         testing::Values(TestData{{1.1_r, 1.2_r, 1.3_r}, {0.1_r, 0.2_r, 0.3_r}},
                                         TestData{{-0.3_r, -0.2_r, -0.1_r},
                                                  {0.7_r, 0.8_r, 0.9_r}}));