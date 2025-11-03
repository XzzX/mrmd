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

#include <Cabana_LinkedCellList.hpp>
#include <Cabana_VerletList.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <cstdint>

namespace mrmd
{
// convenience casting functions for unsigned ints
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr unsigned int uint_c(const T& val)
{
    return static_cast<unsigned int>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr uint8_t uint8_c(const T& val)
{
    return static_cast<uint8_t>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr uint16_t uint16_c(const T& val)
{
    return static_cast<uint16_t>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr uint32_t uint32_c(const T& val)
{
    return static_cast<uint32_t>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr uint64_t uint64_c(const T& val)
{
    return static_cast<uint64_t>(val);
}

// convenience casting functions for signed ints
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr int int_c(const T& val)
{
    return static_cast<int>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr int8_t int8_c(const T& val)
{
    return static_cast<int8_t>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr int16_t int16_c(const T& val)
{
    return static_cast<int16_t>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr int32_t int32_c(const T& val)
{
    return static_cast<int32_t>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr int64_t int64_c(const T& val)
{
    return static_cast<int64_t>(val);
}

using idx_t = int64_t;
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr idx_t idx_c(const T& val)
{
    return static_cast<idx_t>(val);
}

// convenience casting functions for floating point
using real_t = double;
using Point3D = Kokkos::Array<real_t, 3>;
using Vector3D = Kokkos::Array<real_t, 3>;

KOKKOS_INLINE_FUNCTION constexpr real_t operator""_r(const long double val)
{
    return static_cast<real_t>(val);
}

KOKKOS_INLINE_FUNCTION constexpr real_t operator""_r(const unsigned long long val)
{
    return static_cast<real_t>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr real_t real_c(T t)
{
    return static_cast<real_t>(t);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr double double_c(const T& val)
{
    return static_cast<double>(val);
}

template <typename T>
KOKKOS_INLINE_FUNCTION constexpr float float_c(const T& val)
{
    return static_cast<float>(val);
}

enum class AXIS : idx_t
{
    X = 0,  ///< index of the x coordinate in vector views
    Y = 1,  ///< index of the y coordinate in vector views
    Z = 2   ///< index of the z coordinate in vector views
};
// same as C++23 std::to_underlying but with __host__ __device__ annotations
template <class Enum>
KOKKOS_INLINE_FUNCTION constexpr std::underlying_type_t<Enum> to_underlying(Enum e) noexcept
{
    return static_cast<std::underlying_type_t<Enum>>(e);
}

using IndexView = Kokkos::View<idx_t*>;
using IntView = Kokkos::View<idx_t*>;
using IntScatterView = Kokkos::Experimental::ScatterView<idx_t*>;
using PairView = Kokkos::View<idx_t* [2]>;
using SingleView = Kokkos::View<real_t>;
using SingleScatterView = Kokkos::Experimental::ScatterView<real_t>;
using ScalarView = Kokkos::View<real_t*>;
using ScalarScatterView = Kokkos::Experimental::ScatterView<real_t*>;
using MultiView = Kokkos::View<real_t**>;
using MultiScatterView = Kokkos::Experimental::ScatterView<real_t**>;
using VectorView = Kokkos::View<real_t* [3]>;
using VectorScatterView = Kokkos::Experimental::ScatterView<real_t* [3]>;

using HostType = Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
                                Kokkos::DefaultHostExecutionSpace::memory_space>;
using DeviceType =
    Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::memory_space>;
using LinkedCellList = Cabana::LinkedCellList<typename DeviceType::memory_space>;
using VerletList [[deprecated]] = Cabana::VerletList<Kokkos::DefaultExecutionSpace::memory_space,
                                                     Cabana::HalfNeighborTag,
                                                     Cabana::VerletLayout2D,
                                                     Cabana::TeamOpTag>;
using HalfVerletList = Cabana::VerletList<Kokkos::DefaultExecutionSpace::memory_space,
                                          Cabana::HalfNeighborTag,
                                          Cabana::VerletLayout2D,
                                          Cabana::TeamOpTag>;
using FullVerletList = Cabana::VerletList<Kokkos::DefaultExecutionSpace::memory_space,
                                          Cabana::FullNeighborTag,
                                          Cabana::VerletLayout2D,
                                          Cabana::TeamOpTag>;
using NeighborList [[deprecated]] = Cabana::NeighborList<HalfVerletList>;
using HalfNeighborList = Cabana::NeighborList<HalfVerletList>;
using FullNeighborList = Cabana::NeighborList<FullVerletList>;

}  // namespace mrmd