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

mrmd_add_test(NAME test.action.BerendsenBarostat
        FILES BerendsenBarostat.test.cpp)

mrmd_add_test(NAME test.action.BerendsenThermostat
        FILES BerendsenThermostat.test.cpp)

mrmd_add_test(NAME test.action.ContributeMoleculeForceToAtoms
        FILES ContributeMoleculeForceToAtoms.test.cpp)

mrmd_add_test(NAME test.action.CoulombDSF
        FILES CoulombDSF.test.cpp)

mrmd_add_test(NAME test.action.LangevinThermostat
        FILES LangevinThermostat.test.cpp)

mrmd_add_test(NAME test.action.LennardJones
        FILES LennardJones.test.cpp)

mrmd_add_test(NAME test.action.LimitAcceleration
        FILES LimitAcceleration.test.cpp)

mrmd_add_test(NAME test.action.LimitVelocity
        FILES LimitVelocity.test.cpp)

mrmd_add_test(NAME test.action.LJ_IdealGas
        FILES LJ_IdealGas.test.cpp)

mrmd_add_test(NAME test.action.Shake
        FILES Shake.test.cpp)

mrmd_add_test(NAME test.action.SPC
        FILES SPC.test.cpp)

mrmd_add_test(NAME test.action.UpdateMolecules
        FILES UpdateMolecules.test.cpp)

mrmd_add_test(NAME test.action.VelocityVerlet
        FILES VelocityVerlet.test.cpp)