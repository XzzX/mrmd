#include <Kokkos_Core.hpp>

#include "Cabana_AoSoA.hpp"
#include "datatypes.hpp"

using namespace mrmd;

class CabanaVec
{
public:
    constexpr static idx_t size = 10000;
    constexpr static idx_t dim = 3;
    constexpr static int VectorLength = 8;

    using DeviceType = Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;
    using DataTypes = Cabana::MemberTypes<double[dim]>;
    using ParticlesT = Cabana::AoSoA<DataTypes, DeviceType, VectorLength>;

    real_t& get(const idx_t idx, const idx_t dim) const { return Cabana::slice<0>(vec_)(idx, dim); }

private:
    ParticlesT vec_ = ParticlesT("vec", size);
};

class KokkosVec
{
public:
    constexpr static idx_t size = 10000;
    constexpr static idx_t dim = 3;
    using VectorView = Kokkos::View<real_t**>;

    real_t& get(const idx_t idx, const idx_t dim) const { return vec_(dim, idx); }

private:
    VectorView vec_ = VectorView("vec", dim, size);
};

template <typename VECTOR_T>
void native_loop()
{
    VECTOR_T vec;
    for (idx_t idx = 0; idx < VECTOR_T::size; ++idx)
    {
        vec.get(idx, 0) += 5;
        vec.get(idx, 1) += 6;
        vec.get(idx, 2) += 7;
    };
    for (auto i = 0; i < 10; ++i)
    {
        std::cout << vec.get(i, 0) << " | " << vec.get(i, 1) << " | " << vec.get(i, 2) << std::endl;
    }
}

template <typename VECTOR_T>
void kokkos_loop()
{
    VECTOR_T vec;

    const int VECTOR_LENGTH = VECTOR_T::size;
    const int TEAM_SIZE = 1;
    const int NUMBER_OF_TEAMS = 1;

    auto policy = Kokkos::TeamPolicy<Kokkos::OpenMP>(NUMBER_OF_TEAMS, TEAM_SIZE, VECTOR_LENGTH);
    using member_type = Kokkos::TeamPolicy<>::member_type;
    auto kernel = KOKKOS_LAMBDA(member_type teamMember)
    {
        const int e = teamMember.league_rank() * teamMember.team_size() * VECTOR_LENGTH;

        auto vectorPolicy = Kokkos::ThreadVectorRange<>(teamMember, VECTOR_LENGTH);
        auto vectorKernel = KOKKOS_LAMBDA(int idx)
        {
            vec.get(e + idx, 0) += 5;
            //            vec.get(e + idx, 1) += 6;
            //            vec.get(e + idx, 2) += 7;
        };
        Kokkos::parallel_for(vectorPolicy, vectorKernel);
    };
    Kokkos::parallel_for(policy, kernel);

    for (auto i = 0; i < 10; ++i)
    {
        std::cout << vec.get(i, 0) << " | " << vec.get(i, 1) << " | " << vec.get(i, 2) << std::endl;
    }
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    kokkos_loop<KokkosVec>();

    return EXIT_SUCCESS;
}