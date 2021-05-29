cmake_minimum_required(VERSION 3.16)

project(ecab LANGUAGES C CXX)

# set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(external/Cabana)

option(ECAB_LOCAL_ARCHITECTURE "Use instruction set of the local architecture." OFF)
option(ECAB_VEC_REPORT "Enable reporting of loop vectorization." OFF)
option(ECAB_WERROR "Treat warnings as errors." OFF)
add_subdirectory(src)
