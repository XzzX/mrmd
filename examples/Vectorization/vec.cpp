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
