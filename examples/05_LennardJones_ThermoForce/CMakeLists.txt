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

add_executable(05_B02a_LennardJones_ThermodynamicForce
        05_B02a_LennardJones_ThermodynamicForce.cpp
        )
target_link_libraries(05_B02a_LennardJones_ThermodynamicForce
        PRIVATE mrmd CLI11::CLI11
        )
add_executable(05_B02b_LennardJones_ThermodynamicForce_InitialGuess
        05_B02b_LennardJones_ThermodynamicForce_InitialGuess.cpp
        )
target_link_libraries(05_B02b_LennardJones_ThermodynamicForce_InitialGuess
        PRIVATE mrmd CLI11::CLI11
        )

add_executable(05_B03_LennardJones_TracerProduction
        05_B03_LennardJones_TracerProduction.cpp
        )
target_link_libraries(05_B03_LennardJones_TracerProduction
        PRIVATE mrmd CLI11::CLI11
        )