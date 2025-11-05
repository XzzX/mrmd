// Copyright 2024 Sebastian Eibl
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

namespace mrmd::data
{
struct EnergyAndVirialReducer
{
    real_t energy = 0_r;
    real_t virial = 0_r;

    KOKKOS_INLINE_FUNCTION
    EnergyAndVirialReducer() = default;
    KOKKOS_INLINE_FUNCTION
    EnergyAndVirialReducer(const EnergyAndVirialReducer& rhs) = default;
    KOKKOS_INLINE_FUNCTION
    EnergyAndVirialReducer& operator=(const EnergyAndVirialReducer& rhs) = default;

    KOKKOS_INLINE_FUNCTION
    EnergyAndVirialReducer& operator+=(const EnergyAndVirialReducer& src)
    {
        energy += src.energy;
        virial += src.virial;
        return *this;
    }
};
}  // namespace mrmd::data

namespace Kokkos
{  // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<mrmd::data::EnergyAndVirialReducer>
{
    KOKKOS_FORCEINLINE_FUNCTION static mrmd::data::EnergyAndVirialReducer sum()
    {
        return mrmd::data::EnergyAndVirialReducer();
    }
};
}  // namespace Kokkos