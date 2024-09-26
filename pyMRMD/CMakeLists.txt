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

include(CMakePrintHelpers)
find_package(Python3 COMPONENTS Interpreter Development)
cmake_print_variables(Python3_SITEARCH)

FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG a2e59f0e7065404b44dfe92a28aca47ba1378dc4 #2.13.6
)
FetchContent_MakeAvailable(pybind11)

add_library(pyMRMD MODULE
        pyMRMD.cpp
        action.cpp
        analysis.cpp
        cabana.cpp
        communication.cpp
        data.cpp
        io.cpp
        util.cpp
        weighting_function.cpp)
target_link_libraries(pyMRMD PRIVATE mrmd)
target_link_libraries(pyMRMD PRIVATE pybind11::module pybind11::windows_extras)

pybind11_extension(pyMRMD)
pybind11_strip(pyMRMD)

set_target_properties(pyMRMD PROPERTIES CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden")
