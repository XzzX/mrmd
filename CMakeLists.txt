cmake_minimum_required(VERSION 3.16)

project(mrmd LANGUAGES C CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)

include(FetchContent)

FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG 2255619b0f8881de1270054f642accfab25fd597
)
FetchContent_MakeAvailable(CLI11)

get_target_property(TMP CLI11 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(CLI11 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")

FetchContent_Declare(
        Cabana
        GIT_REPOSITORY https://github.com/ECP-copa/Cabana.git
        GIT_TAG 98bc54fcb70c6ea6833095b7688d6b4b4e0a37c5
)
FetchContent_MakeAvailable(Cabana)

get_target_property(TMP cabanacore INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(cabanacore PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0
)
FetchContent_MakeAvailable(googletest)

get_target_property(TMP gtest INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gtest PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")
get_target_property(TMP gmock INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")

enable_testing()

option(MRMD_CLANG_TIDY "Run clang-tidy checks during compilation." OFF)
option(MRMD_LOCAL_ARCHITECTURE "Use instruction set of the local architecture." OFF)
option(MRMD_VEC_REPORT "Enable reporting of loop vectorization." OFF)
option(MRMD_WERROR "Treat warnings as errors." OFF)

if (MRMD_CLANG_TIDY)
    find_program(CLANG_TIDY_COMMAND NAMES clang-tidy)
    if (NOT CLANG_TIDY_COMMAND)
        message(WARNING "MRMD_CLANG_TIDY is ON but clang-tidy is not found!")
        set(CMAKE_CXX_CLANG_TIDY "" CACHE STRING "" FORCE)
    else ()
        configure_file(.clang-tidy .clang-tidy COPYONLY)
        set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_COMMAND})
    endif ()
endif ()

add_subdirectory(src)

add_subdirectory(examples)
add_subdirectory(tests)
