#include <Cabana_Core.hpp>

#include "checks.hpp"
#include <iostream>

void aosoa()
{
    using DataTypes = Cabana::MemberTypes<double[2], double, double>;

    const int VectorLength = 4;

    using MemorySpace = Kokkos::HostSpace;
    using ExecutionSpace = Kokkos::Serial;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

    int num_tuple = 5;
    Cabana::AoSoA<DataTypes, DeviceType, VectorLength> aosoa("my_aosoa", num_tuple);

    std::cout << &Cabana::get<0>(aosoa.access(0), 0, 0) << std::endl;
    std::cout << &Cabana::get<0>(aosoa.access(0), 0, 1) << std::endl;
    std::cout << &Cabana::get<0>(aosoa.access(0), 1, 0) << std::endl;

    CHECK_EQUAL(1, 2, "Hallo");
}

int main(int argc, char* argv[])
{
    Kokkos::ScopeGuard scope_guard(argc, argv);

    aosoa();

    return EXIT_SUCCESS;
}