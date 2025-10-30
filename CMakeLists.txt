# Copyright 2024 Sebastian Eibl
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.25)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(mrmd
        VERSION 0.2.0
        DESCRIPTION "Multi Resolution Molecular Dynamics"
        HOMEPAGE_URL https://github.com/XzzX/mrmd
        LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(TestFunctions)

# set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


option(MRMD_ENABLE_HDF5 "Enable HDF5 support." OFF)
option(MRMD_ENABLE_MPI "Enable MPI support." OFF)
option(MRMD_ENABLE_PYTHON "Enable Python support." OFF)
option(MRMD_ENABLE_COVERAGE "Enable code coverage. (clang)" OFF)
option(MRMD_ENABLE_TESTING "Build tests and add them to ctest." ON)
option(MRMD_VEC_REPORT "Enable reporting of loop vectorization." OFF)
option(MRMD_VERBOSE_ASSERTS "Verbose asserts are only available on CPU!" OFF)
option(MRMD_WERROR "Treat warnings as errors." OFF)

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
if (MRMD_ENABLE_MPI)
    find_package(MPI REQUIRED C CXX)
endif()

########################################################################
# FIND HDF5
########################################################################
if (MRMD_ENABLE_HDF5)
    set(HDF5_PREFER_PARALLEL ON)
    #set(HDF5_FIND_DEBUG ON)
    find_package(HDF5 REQUIRED COMPONENTS C HL)
    if (HDF5_FOUND)
        if (NOT HDF5_IS_PARALLEL)
            message(FATAL_ERROR "HDF5 with MPI support is required! Serial version detected.")
        endif ()
        #need to avoid _FORTIFY_SOURCE=2 which causes trouble in debug builds
        set_target_properties(hdf5::hdf5 PROPERTIES INTERFACE_COMPILE_DEFINITIONS "")
    endif ()
endif()

########################################################################
# FetchContent
########################################################################
include(FetchContent)

FetchContent_Declare(
    Kokkos
    GIT_REPOSITORY https://github.com/kokkos/kokkos.git
    GIT_TAG b601b82d0f4b4491aeafb7b19cde557e92aa761d #4.7.1
    OVERRIDE_FIND_PACKAGE
    SYSTEM
)
FetchContent_MakeAvailable(Kokkos)
include(${kokkos_BINARY_DIR}/KokkosConfigCommon.cmake)

FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG bfffd37e1f804ca4fae1caae106935791696b6a9 #2.6.1
        SYSTEM
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
        YAML_CPP
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG a83cd31548b19d50f3f983b069dceb4f4d50756d #0.8.0+
        SYSTEM
)
FetchContent_MakeAvailable(YAML_CPP)

FetchContent_Declare(
        Cabana
        GIT_REPOSITORY https://github.com/ECP-copa/Cabana.git
        GIT_TAG 7914d28708b1c67b40d5743c1fbc09e221144dd3 #0.7
        SYSTEM
)
FetchContent_MakeAvailable(Cabana)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG 52eb8108c5bdec04579160ae17225d66034bd723 #1.17.0
        SYSTEM
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(mrmd)
if (MRMD_ENABLE_PYTHON)
    add_subdirectory(pyMRMD)
endif()

add_subdirectory(examples)
add_subdirectory(tests)
