#pragma once

#ifdef MRMD_HDF5_AVAILABLE
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

#include "assert/assert.hpp"

namespace mrmd
{
namespace io
{
#ifdef MRMD_HDF5_AVAILABLE
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
#else

using hid_t = int;

template <typename T>
hid_t typeToHDF5()
{
    MRMD_HOST_CHECK(false, "HDF5 support not available!");
    exit(EXIT_FAILURE);
}

template <typename T>
auto CHECK_HDF5(const T& status)
{
    MRMD_HOST_CHECK(false, "HDF5 support not available!");
    exit(EXIT_FAILURE);
}
#endif

}  // namespace io
}  // namespace mrmd