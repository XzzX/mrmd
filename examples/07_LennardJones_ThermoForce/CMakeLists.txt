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

add_executable(07_LennardJones_ThermodynamicForce
        07_LennardJones_ThermodynamicForce.cpp
        )
target_link_libraries(07_LennardJones_ThermodynamicForce
        PRIVATE mrmd CLI11::CLI11
        )
add_executable(07_LennardJones_ThermodynamicForce_InitialGuess
        07_LennardJones_ThermodynamicForce_InitialGuess.cpp
        )
target_link_libraries(07_LennardJones_ThermodynamicForce_InitialGuess
        PRIVATE mrmd CLI11::CLI11
        )

if (MRMD_ENABLE_HDF5)
configure_file(../05_LennardJones_Equilibration/equilibrateLangevin_final.h5md . COPYONLY)
configure_file(thermodynamicForce_initialGuess.txt . COPYONLY)
add_test(NAME Example.07_LennardJones_ThermodynamicForce COMMAND ./07_LennardJones_ThermodynamicForce --nsteps 10)
add_test(NAME Example.07_LennardJones_ThermodynamicForce_InitialGuess COMMAND ./07_LennardJones_ThermodynamicForce_InitialGuess --nsteps 10)
endif ()