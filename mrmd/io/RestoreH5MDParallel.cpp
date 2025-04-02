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

#include "RestoreH5MDParallel.hpp"

#include <fmt/format.h>

#include "assert/assert.hpp"
#include "cmake.hpp"

namespace mrmd::io
{

#ifdef MRMD_ENABLE_HDF5
template <typename T>
void RestoreH5MDParallel::readParallel(hid_t fileId,
                                       const std::string& dataset,
                                       std::vector<T>& data,
                                       const idx_t& saveCount)
{
    auto dset = CHECK_HDF5(H5Dopen(fileId, dataset.c_str(), H5P_DEFAULT));
    auto dspace = CHECK_HDF5(H5Dget_space(dset));

    // get global dimensions
    std::vector<hsize_t> globalDims;
    auto ndims = CHECK_HDF5(H5Sget_simple_extent_ndims(dspace));
    MRMD_HOST_CHECK_GREATER(ndims, 0);
    globalDims.resize(ndims);
    CHECK_HDF5(H5Sget_simple_extent_dims(dspace, globalDims.data(), nullptr));

    // get local dimensions and offset
    std::vector<hsize_t> localDims = globalDims;
    hsize_t localOffset = 0;
    for (auto rk = 0; rk < mpiInfo_->rank; ++rk)
    {
        localOffset += globalDims[1] / uint_c(mpiInfo_->size) +
                       (globalDims[1] % uint_c(mpiInfo_->size) > uint_c(rk) ? 1UL : 0UL);
    }
    auto localSize = globalDims[1] / uint_c(mpiInfo_->size) +
                     (globalDims[1] % uint_c(mpiInfo_->size) > uint_c(mpiInfo_->rank) ? 1UL : 0UL);
    localDims[0] = 1;  // only read one timeframe
    localDims[1] = localSize;

    // set up local part of the input file
    std::vector<hsize_t> offset(globalDims.size(), 0);
    offset[0] = saveCount;
    offset[1] = localOffset;
    std::vector<hsize_t> stride(globalDims.size(), 1);
    std::vector<hsize_t> count(globalDims.size(), 1);
    // check if in bounds
    for (auto i = 0; i < int_c(globalDims.size()); ++i)
    {
        MRMD_HOST_CHECK_LESSEQUAL(
            localDims[i] + offset[i], globalDims[i], fmt::format("i = {}", i));
    }
    auto fileSpace = CHECK_HDF5(H5Dget_space(dset));
    CHECK_HDF5(H5Sselect_hyperslab(
        fileSpace, H5S_SELECT_SET, offset.data(), stride.data(), count.data(), localDims.data()));

    // set up memory data layout
    hid_t memSpace =
        CHECK_HDF5(H5Screate_simple(int_c(localDims.size()), localDims.data(), nullptr));
    auto linLocalSize =
        std::accumulate(localDims.begin(), localDims.end(), hsize_t(1), std::multiplies<>());
    data.resize(linLocalSize);

    // read
    auto dataread = CHECK_HDF5(H5Pcreate(H5P_DATASET_XFER));
    CHECK_HDF5(H5Pset_dxpl_mpio(dataread, H5FD_MPIO_COLLECTIVE));
    CHECK_HDF5(H5Dread(dset, typeToHDF5<T>(), memSpace, fileSpace, dataread, data.data()));

    // close
    CHECK_HDF5(H5Sclose(fileSpace));
    CHECK_HDF5(H5Sclose(memSpace));
    CHECK_HDF5(H5Pclose(dataread));
    CHECK_HDF5(H5Sclose(dspace));
    CHECK_HDF5(H5Dclose(dset));
}

void RestoreH5MDParallel::restore(const std::string& filename,
                                  data::Subdomain& subdomain,
                                  data::Atoms& atoms,
                                  const idx_t& saveCount)
{
    MPI_Info info = MPI_INFO_NULL;

    auto plist = CHECK_HDF5(H5Pcreate(H5P_FILE_ACCESS));
    CHECK_HDF5(H5Pset_fapl_mpio(plist, mpiInfo_->comm, info));

    auto fileId = CHECK_HDF5(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist));

    std::string groupName = "/particles/" + particleSubGroupName_ + "/box";
    CHECK_HDF5(H5LTget_attribute_double(
        fileId, groupName.c_str(), "minCorner", subdomain.minCorner.data()));
    CHECK_HDF5(H5LTget_attribute_double(
        fileId, groupName.c_str(), "maxCorner", subdomain.maxCorner.data()));
    CHECK_HDF5(H5LTget_attribute_double(
        fileId, groupName.c_str(), "ghostLayerThickness", subdomain.ghostLayerThickness.data()));
    subdomain =
        data::Subdomain(subdomain.minCorner, subdomain.maxCorner, subdomain.ghostLayerThickness);
    std::vector<real_t> pos;
    if (restorePos)
    {
        readParallel(fileId,
                     "/particles/" + particleSubGroupName_ + "/" + posDataset + "/value",
                     pos,
                     saveCount);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 3, pos.size());
    }
    std::vector<real_t> vel;
    if (restoreVel)
    {
        readParallel(fileId,
                     "/particles/" + particleSubGroupName_ + "/" + velDataset + "/value",
                     vel,
                     saveCount);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 3, vel.size());
    }
    std::vector<real_t> force;
    if (restoreForce)
    {
        readParallel(fileId,
                     "/particles/" + particleSubGroupName_ + "/" + forceDataset + "/value",
                     force,
                     saveCount);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 3, force.size());
    }
    std::vector<idx_t> type;
    if (restoreType)
    {
        readParallel(fileId,
                     "/particles/" + particleSubGroupName_ + "/" + typeDataset + "/value",
                     type,
                     saveCount);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 1, type.size());
    }
    std::vector<real_t> mass;
    if (restoreMass)
    {
        readParallel(fileId,
                     "/particles/" + particleSubGroupName_ + "/" + massDataset + "/value",
                     mass,
                     saveCount);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 1, mass.size());
    }
    std::vector<real_t> charge;
    if (restoreCharge)
    {
        readParallel(fileId,
                     "/particles/" + particleSubGroupName_ + "/" + chargeDataset + "/value",
                     charge,
                     saveCount);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 1, charge.size());
    }
    std::vector<real_t> relativeMass;
    if (restoreRelativeMass)
    {
        readParallel(fileId,
                     "/particles/" + particleSubGroupName_ + "/" + relativeMassDataset + "/value",
                     relativeMass,
                     saveCount);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 1, relativeMass.size());
    }

    idx_t numLocalAtoms = idx_c(pos.size() / 3);
    data::HostAtoms h_atoms(numLocalAtoms);
    h_atoms.numLocalAtoms = numLocalAtoms;
    h_atoms.numGhostAtoms = 0;
    for (idx_t idx = 0; idx < numLocalAtoms; ++idx)
    {
        if (restorePos)
        {
            h_atoms.getPos()(idx, 0) = pos[idx * 3 + 0];
            h_atoms.getPos()(idx, 1) = pos[idx * 3 + 1];
            h_atoms.getPos()(idx, 2) = pos[idx * 3 + 2];
        }
        if (restoreVel)
        {
            h_atoms.getVel()(idx, 0) = vel[idx * 3 + 0];
            h_atoms.getVel()(idx, 1) = vel[idx * 3 + 1];
            h_atoms.getVel()(idx, 2) = vel[idx * 3 + 2];
        }
        if (restoreForce)
        {
            h_atoms.getForce()(idx, 0) = force[idx * 3 + 0];
            h_atoms.getForce()(idx, 1) = force[idx * 3 + 1];
            h_atoms.getForce()(idx, 2) = force[idx * 3 + 2];
        }
        if (restoreType) h_atoms.getType()(idx) = type[idx];
        if (restoreMass) h_atoms.getMass()(idx) = mass[idx];
        if (restoreCharge) h_atoms.getCharge()(idx) = charge[idx];
        if (restoreRelativeMass) h_atoms.getRelativeMass()(idx) = relativeMass[idx];
    }
    data::deep_copy(atoms, h_atoms);

    CHECK_HDF5(H5Fclose(fileId));
}
#else
template <typename T>
void RestoreH5MDParallel::readParallel(hid_t /*fileId*/,
                                       const std::string& /*dataset*/,
                                       std::vector<T>& /*data*/)
{
    MRMD_HOST_CHECK(false, "HDF5 support not available!");
    exit(EXIT_FAILURE);
}

void RestoreH5MDParallel::restore(const std::string& /*filename*/,
                                  data::Subdomain& /*subdomain*/,
                                  data::Atoms& /*atoms*/,
                                  const idx_t& /*step*/)
{
    MRMD_HOST_CHECK(false, "HDF5 support not available!");
    exit(EXIT_FAILURE);
}
#endif

}  // namespace mrmd::io