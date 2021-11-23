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
    KOKKOS_INLINE_FUNCTION
    void operator+=(const volatile EnergyAndVirialReducer& src) volatile
    {
        energy += src.energy;
        virial += src.virial;
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