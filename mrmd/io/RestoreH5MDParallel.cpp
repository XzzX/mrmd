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

#include <numeric>

#include "assert/assert.hpp"
#include "cmake.hpp"

namespace mrmd::io
{

#ifdef MRMD_ENABLE_HDF5
template <typename T>
void RestoreH5MDParallel::read(hid_t fileId,
                               const std::string& dataset,
                               std::vector<T>& data)
{
    auto dset = CHECK_HDF5(H5Dopen(fileId, dataset.c_str(), H5P_DEFAULT));
    auto dspace = CHECK_HDF5(H5Dget_space(dset));

    auto ndims = CHECK_HDF5(H5Sget_simple_extent_ndims(dspace));
    MRMD_HOST_CHECK_GREATER(ndims, 0);
    std::vector<hsize_t> dims(ndims);
    CHECK_HDF5(H5Sget_simple_extent_dims(dspace, dims.data(), nullptr));

    auto totalSize =
        std::accumulate(dims.begin(), dims.end(), hsize_t(1), std::multiplies<>());
    data.resize(totalSize);

    CHECK_HDF5(H5Dread(dset, typeToHDF5<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()));

    CHECK_HDF5(H5Sclose(dspace));
    CHECK_HDF5(H5Dclose(dset));
}

void RestoreH5MDParallel::restore(const std::string& filename, data::Subdomain& subdomain, data::Atoms& atoms)
{
    auto fileId = CHECK_HDF5(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));

    std::string groupName = "/particles/" + particleGroupName_ + "/box";
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
        read(fileId, "/particles/" + particleGroupName_ + "/" + posDataset + "/value", pos);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 3, pos.size());
    }
    std::vector<real_t> vel;
    if (restoreVel)
    {
        read(fileId, "/particles/" + particleGroupName_ + "/" + velDataset + "/value", vel);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 3, vel.size());
    }
    std::vector<real_t> force;
    if (restoreForce)
    {
        read(fileId, "/particles/" + particleGroupName_ + "/" + forceDataset + "/value", force);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 3, force.size());
    }
    std::vector<idx_t> type;
    if (restoreType)
    {
        read(fileId, "/particles/" + particleGroupName_ + "/" + typeDataset + "/value", type);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 1, type.size());
    }
    std::vector<real_t> mass;
    if (restoreMass)
    {
        read(fileId, "/particles/" + particleGroupName_ + "/" + massDataset + "/value", mass);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 1, mass.size());
    }
    std::vector<real_t> charge;
    if (restoreCharge)
    {
        read(fileId, "/particles/" + particleGroupName_ + "/" + chargeDataset + "/value", charge);
        MRMD_HOST_CHECK_EQUAL(pos.size() / 3 * 1, charge.size());
    }
    std::vector<real_t> relativeMass;
    if (restoreRelativeMass)
    {
        read(fileId, "/particles/" + particleGroupName_ + "/" + relativeMassDataset + "/value", relativeMass);
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
void RestoreH5MDParallel::read(hid_t /*fileId*/,
                               const std::string& /*dataset*/,
                               std::vector<T>& /*data*/)
{
    MRMD_HOST_CHECK(false, "HDF5 support not available!");
    exit(EXIT_FAILURE);
}

void RestoreH5MDParallel::restore(const std::string& /*filename*/, data::Subdomain& /*subdomain*/, data::Atoms& /*atoms*/)
{
    MRMD_HOST_CHECK(false, "HDF5 support not available!");
    exit(EXIT_FAILURE);
}
#endif

}  // namespace mrmd::io