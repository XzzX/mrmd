#include "RestoreH5MDParallel.hpp"

#include <fmt/format.h>

#include "assert.hpp"

namespace mrmd::io
{

template <typename T>
void RestoreH5MDParallel::readParallel(hid_t fileId,
                                       const std::string& dataset,
                                       std::vector<T>& data)
{
    auto dset = CHECK_HDF5(H5Dopen(fileId, dataset.c_str(), H5P_DEFAULT));
    auto dspace = CHECK_HDF5(H5Dget_space(dset));

    // get global dimensions
    std::vector<hsize_t> globalDims;
    auto ndims = CHECK_HDF5(H5Sget_simple_extent_ndims(dspace));
    CHECK_GREATER(ndims, 0);
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
    offset[1] = localOffset;
    std::vector<hsize_t> stride(globalDims.size(), 1);
    std::vector<hsize_t> count(globalDims.size(), 1);
    // check if in bounds
    for (auto i = 0; i < int_c(globalDims.size()); ++i)
    {
        CHECK_LESSEQUAL(localDims[i] + offset[i], globalDims[i], fmt::format("i = {}", i));
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

void RestoreH5MDParallel::restore(const std::string& filename, data::Atoms& atoms)
{
    MPI_Info info = MPI_INFO_NULL;

    auto plist = CHECK_HDF5(H5Pcreate(H5P_FILE_ACCESS));
    CHECK_HDF5(H5Pset_fapl_mpio(plist, mpiInfo_->comm, info));

    auto fileId = CHECK_HDF5(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist));
    std::vector<real_t> pos;
    if (restorePos)
    {
        readParallel(fileId, "/particles/" + particleGroupName_ + "/" + posDataset + "/value", pos);
        CHECK_EQUAL(pos.size() / 3 * 3, pos.size());
    }
    std::vector<real_t> vel;
    if (restoreVel)
    {
        readParallel(fileId, "/particles/" + particleGroupName_ + "/" + velDataset + "/value", vel);
        CHECK_EQUAL(pos.size() / 3 * 3, vel.size());
    }
    std::vector<real_t> force;
    if (restoreForce)
    {
        readParallel(
            fileId, "/particles/" + particleGroupName_ + "/" + forceDataset + "/value", force);
        CHECK_EQUAL(pos.size() / 3 * 3, force.size());
    }
    std::vector<idx_t> type;
    if (restoreType)
    {
        readParallel(
            fileId, "/particles/" + particleGroupName_ + "/" + typeDataset + "/value", type);
        CHECK_EQUAL(pos.size() / 3 * 1, type.size());
    }
    std::vector<real_t> mass;
    if (restoreMass)
    {
        readParallel(
            fileId, "/particles/" + particleGroupName_ + "/" + massDataset + "/value", mass);
        CHECK_EQUAL(pos.size() / 3 * 1, mass.size());
    }
    std::vector<real_t> charge;
    if (restoreCharge)
    {
        readParallel(
            fileId, "/particles/" + particleGroupName_ + "/" + chargeDataset + "/value", charge);
        CHECK_EQUAL(pos.size() / 3 * 1, charge.size());
    }
    std::vector<real_t> relativeMass;
    if (restoreRelativeMass)
    {
        readParallel(fileId,
                     "/particles/" + particleGroupName_ + "/" + relativeMassDataset + "/value",
                     relativeMass);
        CHECK_EQUAL(pos.size() / 3 * 1, relativeMass.size());
    }

    idx_t numLocalAtoms = idx_c(pos.size() / 3);
    atoms.resize(numLocalAtoms);
    atoms.numLocalAtoms = numLocalAtoms;
    atoms.numGhostAtoms = 0;
    for (idx_t idx = 0; idx < numLocalAtoms; ++idx)
    {
        if (restorePos)
        {
            atoms.getPos()(idx, 0) = pos[idx * 3 + 0];
            atoms.getPos()(idx, 1) = pos[idx * 3 + 1];
            atoms.getPos()(idx, 2) = pos[idx * 3 + 2];
        }
        if (restoreVel)
        {
            atoms.getVel()(idx, 0) = vel[idx * 3 + 0];
            atoms.getVel()(idx, 1) = vel[idx * 3 + 1];
            atoms.getVel()(idx, 2) = vel[idx * 3 + 2];
        }
        if (restoreForce)
        {
            atoms.getForce()(idx, 0) = force[idx * 3 + 0];
            atoms.getForce()(idx, 1) = force[idx * 3 + 1];
            atoms.getForce()(idx, 2) = force[idx * 3 + 2];
        }
        if (restoreType) atoms.getType()(idx) = type[idx];
        if (restoreMass) atoms.getMass()(idx) = mass[idx];
        if (restoreCharge) atoms.getCharge()(idx) = charge[idx];
        if (restoreRelativeMass) atoms.getRelativeMass()(idx) = relativeMass[idx];
    }

    CHECK_HDF5(H5Fclose(fileId));
}

}  // namespace mrmd::io