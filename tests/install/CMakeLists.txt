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

# Standalone project that only consumes an *installed* mrmd (via find_package).
# It is intentionally not added as a subdirectory of the main build - it exists
# to verify that `cmake --install` produces a usable mrmdConfig.cmake package.

cmake_minimum_required(VERSION 3.25)

project(mrmd_install_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(mrmd REQUIRED)

include(FetchContent)
FetchContent_Declare(
        CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG bfffd37e1f804ca4fae1caae106935791696b6a9 #2.6.1
        SYSTEM
)
FetchContent_MakeAvailable(CLI11)

add_executable(00_IdealGas_NVE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../examples/00_IdealGas_NVE/00_IdealGas_NVE.cpp)
target_link_libraries(00_IdealGas_NVE
        PRIVATE mrmd::mrmd CLI11::CLI11)
