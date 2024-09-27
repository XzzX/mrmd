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

add_executable(Argon
        Argon.cpp
        )
target_link_libraries(Argon
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )
add_test(NAME Example.Argon COMMAND ./Argon --nsteps 10)
if (MRMD_ENABLE_PYTHON)
    add_test(NAME Example.Argon.Python COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/argon.py --nsteps 10)
    set_property(TEST Example.Argon.Python PROPERTY 
        ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${CMAKE_BUILD_DIR}/pyMRMD")
endif()

add_executable(MultiResArgon
        MultiResArgon.cpp
        )
target_link_libraries(MultiResArgon
        PRIVATE mrmd CLI11::CLI11 fmt::fmt
        )
add_test(NAME Example.MultiResArgon COMMAND ./MultiResArgon --nsteps 10)
if (MRMD_ENABLE_PYTHON)
    add_test(NAME Example.MultiResArgon.Python COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/multi_res_argon.py --nsteps 10)
    set_property(TEST Example.MultiResArgon.Python PROPERTY 
        ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${CMAKE_BUILD_DIR}/pyMRMD")
endif()
