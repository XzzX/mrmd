# Copyright 2024 Sebastian Eibl
# Copyright 2026 Julian Friedrich Hille
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

add_executable(06_LennardJones_ProductionAtomistic
        06_LennardJones_ProductionAtomistic.cpp
        )
target_link_libraries(06_LennardJones_ProductionAtomistic
        PRIVATE mrmd CLI11::CLI11
        )

if (MRMD_ENABLE_HDF5)
configure_file(../05_LennardJones_Equilibration/equilibrateLangevin_final.h5md . COPYONLY)
add_test(NAME Example.06_LennardJones_ProductionAtomistic COMMAND ./06_LennardJones_ProductionAtomistic --nsteps 10)
endif ()