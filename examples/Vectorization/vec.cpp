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

#include <Kokkos_Core.hpp>
#include <iostream>

#include "Cabana_AoSoA.hpp"
#include "datatypes.hpp"

using namespace mrmd;

void loop()
{
    constexpr int LENGTH = 10000;
    ScalarView vec("vec", LENGTH);

    const int VECTOR_LENGTH = LENGTH;
    const int TEAM_SIZE = 1;
    const int NUMBER_OF_TEAMS = 1;

    auto policy = Kokkos::TeamPolicy<>(NUMBER_OF_TEAMS, TEAM_SIZE, VECTOR_LENGTH);
    using member_type = Kokkos::TeamPolicy<>::member_type;
    auto kernel = KOKKOS_LAMBDA(member_type teamMember)
    {
        const int e = teamMember.league_rank() * teamMember.team_size() * VECTOR_LENGTH;

        auto vectorPolicy = Kokkos::ThreadVectorRange<>(teamMember, VECTOR_LENGTH);
        auto vectorKernel = [=](int idx) { vec(e + idx) += 5; };
        Kokkos::parallel_for(vectorPolicy, vectorKernel);
    };
    Kokkos::parallel_for(policy, kernel);

    for (auto i = 0; i < 10; ++i)
    {
        std::cout << vec(i) << std::endl;
    }
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    loop();

    return EXIT_SUCCESS;
}
