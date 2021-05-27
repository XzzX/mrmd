add_executable(ecab main.cpp)
target_link_libraries(ecab PRIVATE Kokkos::kokkos Cabana::cabanacore)

if(ECAB_WERROR)
    target_compile_options(ecab PRIVATE "-Werror")
endif()
# compiler specific configurations
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    target_compile_options(ecab PRIVATE -Wall -Wextra -Wpedantic)
endif()
