cmake_minimum_required(VERSION 3.19)

project(mrmd
        VERSION 0.2.0
        DESCRIPTION "Multi Resolution Molecular Dynamics"
        HOMEPAGE_URL https://github.com/XzzX/mrmd
        LANGUAGES C CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# custom build type DebugOptimized
set(CMAKE_C_FLAGS_DEBUGOPTIMIZED "${CMAKE_C_FLAGS_DEBUG} -O3" CACHE STRING "C flags for DebugOptimized")
set(CMAKE_CXX_FLAGS_DEBUGOPTIMIZED "${CMAKE_CXX_FLAGS_DEBUG} -O3" CACHE STRING "CXX flags for DebugOptimized")
set(CMAKE_EXE_LINKER_FLAGS_DEBUGOPTIMIZED ${CMAKE_EXE_LINKER_FLAGS_DEBUG} CACHE STRING "")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUGOPTIMIZED ${CMAKE_SHARED_LINKER_FLAGS_DEBUG} CACHE STRING "")
set(CMAKE_MODULE_LINKER_FLAGS_DEBUGOPTIMIZED ${CMAKE_MODULE_LINKER_FLAGS_DEBUG} CACHE STRING "")

mark_as_advanced(
        CMAKE_C_FLAGS_DEBUGOPTIMIZED
        CMAKE_CXX_FLAGS_DEBUGOPTIMIZED
        CMAKE_EXE_LINKER_FLAGS_DEBUGOPTIMIZED
        CMAKE_SHARED_LINKER_FLAGS_DEBUGOPTIMIZED
        CMAKE_MODULE_LINKER_FLAGS_DEBUGOPTIMIZED
)

########################################################################
# FIND MPI
########################################################################
find_package(MPI REQUIRED CXX)

########################################################################
# FIND Kokkos
########################################################################
find_package(Kokkos REQUIRED)

########################################################################
# FIND HDF5
########################################################################
set(HDF5_PREFER_PARALLEL ON)
#set(HDF5_FIND_DEBUG ON)
find_package(HDF5 REQUIRED COMPONENTS C HL)
if (NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "HDF5 with MPI support is required! Serial version detected.")
endif ()
#need to avoid _FORTIFY_SOURCE=2 which causes trouble in debug builds
set_target_properties(hdf5::hdf5 PROPERTIES INTERFACE_COMPILE_DEFINITIONS "")

include(FetchContent)

FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG v2.1.2
)
FetchContent_MakeAvailable(CLI11)

get_target_property(TMP CLI11 INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(CLI11 PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")

FetchContent_Declare(
        FMT
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 8.1.0
)
FetchContent_MakeAvailable(FMT)

get_target_property(TMP fmt INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(fmt PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")

FetchContent_Declare(
        YAML_CPP
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG yaml-cpp-0.7.0
)
FetchContent_MakeAvailable(YAML_CPP)

get_target_property(TMP yaml-cpp INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(yaml-cpp PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")

FetchContent_Declare(
        Cabana
        GIT_REPOSITORY https://github.com/ECP-copa/Cabana.git
        GIT_TAG 0.4.0
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
get_target_property(TMP gtest_main INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gtest_main PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")
get_target_property(TMP gmock INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gmock PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")
get_target_property(TMP gmock_main INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(gmock_main PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${TMP}")

enable_testing()

option(MRMD_CLANG_TIDY "Run clang-tidy checks during compilation." OFF)
option(MRMD_LOCAL_ARCHITECTURE "Use instruction set of the local architecture." OFF)
option(MRMD_VEC_REPORT "Enable reporting of loop vectorization." OFF)
option(MRMD_VERBOSE_ASSERTS "Verbose asserts are only available on CPU!" OFF)
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
