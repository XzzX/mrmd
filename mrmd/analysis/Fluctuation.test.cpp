// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "Fluctuation.hpp"

#include <gtest/gtest.h>

namespace mrmd
{
namespace analysis
{
void fluctuationNormalization()
{
    auto histogram0 = data::MultiHistogram("hist", 0_r, 1_r, 10, 1);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram0.data(idx, 0) = 3_r;
        });
    Kokkos::fence();
    auto fluctuation0 = analysis::getFluctuation(histogram0, 2_r, 0);

    auto histogram1 = data::MultiHistogram("hist", 0_r, 2_r, 10, 1);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram1.data(idx, 0) = 3_r;
        });
    Kokkos::fence();
    auto fluctuation1 = analysis::getFluctuation(histogram1, 2_r, 0);

    EXPECT_FLOAT_EQ(fluctuation0, fluctuation1);
}
TEST(Fluctuation, normalization) { fluctuationNormalization(); }

void fluctuationCheck()
{
    auto histogram = data::MultiHistogram("hist", 0_r, 2_r, 10, 2);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = 3_r;
            histogram.data(idx, 1) = 4_r;
        });
    Kokkos::fence();

    auto fluctuation0 = analysis::getFluctuation(histogram, 2_r, 0);
    auto fluctuation1 = analysis::getFluctuation(histogram, 2_r, 1);

    EXPECT_FLOAT_EQ(fluctuation0, 0.25_r);
    EXPECT_FLOAT_EQ(fluctuation1, 1_r);
}
TEST(Fluctuation, check) { fluctuationCheck(); }

void fluctuationRelation()
{
    auto histogram = data::MultiHistogram("hist", 0_r, 1_r, 10, 2);
    Kokkos::parallel_for(
        "init_hist", Kokkos::RangePolicy<>(0, 10), KOKKOS_LAMBDA(const idx_t idx) {
            histogram.data(idx, 0) = real_c(idx + 1);
            histogram.data(idx, 1) = real_c(idx);
        });
    Kokkos::fence();

    auto fluctuation0 = analysis::getFluctuation(histogram, 0.5_r, 0);
    auto fluctuation1 = analysis::getFluctuation(histogram, 0.5_r, 1);

    EXPECT_GT(fluctuation0, fluctuation1);
}
TEST(Fluctuation, relation) { fluctuationRelation(); }
}  // namespace analysis
}  // namespace mrmd