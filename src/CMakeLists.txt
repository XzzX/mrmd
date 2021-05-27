add_executable(ecab main.cpp)
target_link_libraries(ecab PRIVATE Kokkos::kokkos Cabana::cabanacore)
