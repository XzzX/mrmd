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

#ifdef MRMD_ENABLE_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

#include "assert/assert.hpp"

namespace mrmd
{
namespace io
{
#ifdef MRMD_ENABLE_HDF5
template <typename T>
hid_t typeToHDF5()
{
    MRMD_HOST_CHECK(false, "This type is not supported!");
    exit(EXIT_FAILURE);
}
template <>
inline hid_t typeToHDF5<int8_t>()
{
    return H5T_NATIVE_INT8;
}
template <>
inline hid_t typeToHDF5<int16_t>()
{
    return H5T_NATIVE_INT16;
}
template <>
inline hid_t typeToHDF5<int32_t>()
{
    return H5T_NATIVE_INT32;
}
template <>
inline hid_t typeToHDF5<int64_t>()
{
    return H5T_NATIVE_INT64;
}
template <>
inline hid_t typeToHDF5<double>()
{
    return H5T_NATIVE_DOUBLE;
}

template <typename T>
auto CHECK_HDF5(const T& status)
{
    if (status < 0)
    {
        H5Eprint(H5E_DEFAULT, stderr);
        MRMD_HOST_CHECK_GREATEREQUAL(status, 0);
    }
    return status;
}
#endif

}  // namespace io
}  // namespace mrmd