cmake_minimum_required(VERSION 3.16)

project(ecab LANGUAGES C CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

find_package(Kokkos REQUIRED)

option(ECAB_LOCAL_ARCHITECTURE "Use instruction set of the local architecture." OFF)
option(ECAB_VEC_REPORT "Enable reporting of loop vectorization." OFF)
option(ECAB_WERROR "Treat warnings as errors." OFF)
add_subdirectory(src)

add_subdirectory(examples)
add_subdirectory(tests)

include(FetchContent)

FetchContent_Declare(
        Cabana
        GIT_REPOSITORY https://github.com/ECP-copa/Cabana.git
        GIT_TAG 98bc54fcb70c6ea6833095b7688d6b4b4e0a37c5
)
FetchContent_MakeAvailable(Cabana)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.10.0
)
FetchContent_MakeAvailable(googletest)
