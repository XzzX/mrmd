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

#include "DumpH5MDParallel.hpp"

#include <fmt/format.h>

#include <numeric>

#include "assert/assert.hpp"
#ifdef MRMD_ENABLE_HDF5
#include "hdf5.hpp"
#endif
#include "version.hpp"

namespace mrmd::io
{

#ifdef MRMD_ENABLE_HDF5
namespace impl
{
/**
 * This class collects everything needed for the HDF5 dump. If HDF5 is disabled,
 * this class can be skipped and all HDF5 dependencies are gone.
 */
class DumpH5MDParallelImpl
{
public:
    explicit DumpH5MDParallelImpl(DumpH5MDParallel& config) : config_(config) {}

    void open(const std::string& filename, const data::Atoms& atoms);
    void dumpStep(const data::Subdomain& subdomain,
                  const data::Atoms& atoms,
                  const idx_t step,
                  const real_t dt);
    void close() const;

    void dump(const std::string& filename,
              const data::Subdomain& subdomain,
              const data::Atoms& atoms);

private:
    hid_t createFile(const std::string& filename, const hid_t& propertyList) const;
    void closeFile(const hid_t& fileId) const;
    hid_t createGroup(const hid_t& parentElementId, const std::string& groupName) const;
    void closeGroup(const hid_t& groupId) const;
    void openBox() const;
    hid_t createChunkedDataset(const hid_t& groupId,
                               const std::vector<hsize_t>& dims,
                               const std::string& name,
                               const hid_t& dtype) const;
    void closeDataset(const hid_t& datasetId) const;

    template <typename T>
    void appendData(const hid_t datasetId,
                    const std::vector<T>& data,
                    const std::vector<hsize_t>& dims) const;
    template <typename T>
    void appendDataParallel(const hid_t datasetId,
                            const std::vector<T>& data,
                            const std::vector<hsize_t>& dims) const;
    void appendEdges(const idx_t& step, const real_t& dt, const data::Subdomain& subdomain) const;
    void appendCharges(const idx_t& step, const real_t& dt, const data::HostAtoms& atoms) const;
    void appendForces(const idx_t& step, const real_t& dt, const data::HostAtoms& atoms) const;
    void appendMasses(const idx_t& step, const real_t& dt, const data::HostAtoms& atoms) const;
    void appendPositions(const idx_t& step, const real_t& dt, const data::HostAtoms& atoms) const;
    void appendRelativeMasses(const idx_t& step,
                              const real_t& dt,
                              const data::HostAtoms& atoms) const;
    void appendTypes(const idx_t& step, const real_t& dt, const data::HostAtoms& atoms) const;
    void appendVelocities(const idx_t& step, const real_t& dt, const data::HostAtoms& atoms) const;

    void updateCache(const data::HostAtoms& atoms);

    void writeHeader(hid_t fileId) const;
    void writeBox(hid_t fileId, const data::Subdomain& subdomain) const;
    void writePos(hid_t fileId, const data::HostAtoms& atoms);
    void writeVel(hid_t fileId, const data::HostAtoms& atoms);
    void writeForce(hid_t fileId, const data::HostAtoms& atoms);
    void writeType(hid_t fileId, const data::HostAtoms& atoms);
    void writeMass(hid_t fileId, const data::HostAtoms& atoms);
    void writeCharge(hid_t fileId, const data::HostAtoms& atoms);
    void writeRelativeMass(hid_t fileId, const data::HostAtoms& atoms);

    template <typename T>
    void writeParallel(hid_t fileId,
                       const std::string& name,
                       const std::vector<hsize_t>& globalDims,
                       const std::vector<hsize_t>& localDims,
                       const std::vector<T>& data);

    DumpH5MDParallel& config_;

    int64_t numLocalParticles = -1;
    int64_t numTotalParticles = -1;
    /// Offset of the local particle chunk in the global particle array.
    int64_t particleOffset = -1;
};

void DumpH5MDParallelImpl::open(const std::string& filename, const data::Atoms& atoms)
{
    MPI_Info info = MPI_INFO_NULL;

    auto propertyList = CHECK_HDF5(H5Pcreate(H5P_FILE_ACCESS));
    CHECK_HDF5(H5Pset_fapl_mpio(propertyList, config_.mpiInfo->comm, info));

    config_.fileId = createFile(filename, propertyList);

    CHECK_HDF5(H5Pclose(propertyList));

    config_.particleGroupId = createGroup(config_.fileId, "particles");
    config_.particleSubGroupId = createGroup(config_.particleGroupId, config_.particleSubGroupName);
    writeHeader(config_.fileId);
    openBox();

    config_.chargesGroupId = createGroup(config_.particleSubGroupId, "charge");
    config_.chargesStepSetId = createChunkedDataset(
        config_.chargesGroupId, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    config_.chargesTimeSetId = createChunkedDataset(
        config_.chargesGroupId, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    config_.chargesValueSetId = createChunkedDataset(config_.chargesGroupId,
                                                     std::vector<hsize_t>{1, atoms.size(), 1},
                                                     "value",
                                                     H5T_NATIVE_DOUBLE);

    config_.forceGroupId = createGroup(config_.particleSubGroupId, "force");
    config_.forceStepSetId = createChunkedDataset(
        config_.forceGroupId, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    config_.forceTimeSetId = createChunkedDataset(
        config_.forceGroupId, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    config_.forceValueSetId = createChunkedDataset(
        config_.forceGroupId, std::vector<hsize_t>{1, atoms.size(), 3}, "value", H5T_NATIVE_DOUBLE);

    config_.massGroupId = createGroup(config_.particleSubGroupId, "mass");
    config_.massStepSetId = createChunkedDataset(
        config_.massGroupId, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    config_.massTimeSetId = createChunkedDataset(
        config_.massGroupId, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    config_.massValueSetId = createChunkedDataset(
        config_.massGroupId, std::vector<hsize_t>{1, atoms.size(), 1}, "value", H5T_NATIVE_DOUBLE);

    config_.posGroupId = createGroup(config_.particleSubGroupId, "position");
    config_.posStepSetId =
        createChunkedDataset(config_.posGroupId, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    config_.posTimeSetId = createChunkedDataset(
        config_.posGroupId, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    config_.posValueSetId = createChunkedDataset(
        config_.posGroupId, std::vector<hsize_t>{1, atoms.size(), 3}, "value", H5T_NATIVE_DOUBLE);

    config_.relativeMassGroupId = createGroup(config_.particleSubGroupId, "relativeMass");
    config_.relativeMassStepSetId = createChunkedDataset(
        config_.relativeMassGroupId, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    config_.relativeMassTimeSetId = createChunkedDataset(
        config_.relativeMassGroupId, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    config_.relativeMassValueSetId = createChunkedDataset(config_.relativeMassGroupId,
                                                          std::vector<hsize_t>{1, atoms.size(), 1},
                                                          "value",
                                                          H5T_NATIVE_DOUBLE);

    config_.typeGroupId = createGroup(config_.particleSubGroupId, "type");
    config_.typeStepSetId = createChunkedDataset(
        config_.typeGroupId, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    config_.typeTimeSetId = createChunkedDataset(
        config_.typeGroupId, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    config_.typeValueSetId = createChunkedDataset(
        config_.typeGroupId, std::vector<hsize_t>{1, atoms.size(), 1}, "value", H5T_NATIVE_INT64);

    config_.velGroupId = createGroup(config_.particleSubGroupId, "velocity");
    config_.velStepSetId =
        createChunkedDataset(config_.velGroupId, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    config_.velTimeSetId = createChunkedDataset(
        config_.velGroupId, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    config_.velValueSetId = createChunkedDataset(
        config_.velGroupId, std::vector<hsize_t>{1, atoms.size(), 3}, "value", H5T_NATIVE_DOUBLE);
}

hid_t DumpH5MDParallelImpl::createFile(const std::string& filename, const hid_t& propertyList) const
{
    auto fileId = CHECK_HDF5(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, propertyList));
    return fileId;
}

void DumpH5MDParallelImpl::closeFile(const hid_t& fileId) const { CHECK_HDF5(H5Fclose(fileId)); }

hid_t DumpH5MDParallelImpl::createGroup(const hid_t& parentElementId,
                                        const std::string& groupName) const
{
    auto groupId = CHECK_HDF5(
        H5Gcreate(parentElementId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    return groupId;
}

void DumpH5MDParallelImpl::closeGroup(const hid_t& groupId) const { CHECK_HDF5(H5Gclose(groupId)); }

void DumpH5MDParallelImpl::close() const
{
    closeDataset(config_.velValueSetId);
    closeDataset(config_.velTimeSetId);
    closeDataset(config_.velStepSetId);
    closeGroup(config_.velGroupId);
    closeDataset(config_.typeValueSetId);
    closeDataset(config_.typeTimeSetId);
    closeDataset(config_.typeStepSetId);
    closeGroup(config_.typeGroupId);
    closeDataset(config_.relativeMassValueSetId);
    closeDataset(config_.relativeMassTimeSetId);
    closeDataset(config_.relativeMassStepSetId);
    closeGroup(config_.relativeMassGroupId);
    closeDataset(config_.posValueSetId);
    closeDataset(config_.posTimeSetId);
    closeDataset(config_.posStepSetId);
    closeGroup(config_.posGroupId);
    closeDataset(config_.massStepSetId);
    closeDataset(config_.massTimeSetId);
    closeDataset(config_.massValueSetId);
    closeGroup(config_.massGroupId);
    closeDataset(config_.chargesValueSetId);
    closeDataset(config_.chargesTimeSetId);
    closeDataset(config_.chargesStepSetId);
    closeDataset(config_.forceValueSetId);
    closeDataset(config_.forceTimeSetId);
    closeDataset(config_.forceStepSetId);
    closeGroup(config_.forceGroupId);
    closeGroup(config_.chargesGroupId);
    closeDataset(config_.edgesValueSetId);
    closeDataset(config_.edgesTimeSetId);
    closeDataset(config_.edgesStepSetId);
    closeGroup(config_.edgesGroupId);
    closeGroup(config_.boxGroupId);
    closeGroup(config_.particleSubGroupId);
    closeGroup(config_.particleGroupId);
    closeFile(config_.fileId);
}

void DumpH5MDParallelImpl::openBox() const
{
    config_.boxGroupId = createGroup(config_.particleSubGroupId, "box");

    std::vector<int> dims = {3};
    CHECK_HDF5(H5LTset_attribute_int(
        config_.particleSubGroupId, "box", "dimension", dims.data(), dims.size()));

    auto boundaryType = H5Tcopy(H5T_C_S1);
    CHECK_HDF5(H5Tset_size(boundaryType, 8));
    CHECK_HDF5(H5Tset_strpad(boundaryType, H5T_STR_NULLPAD));
    std::vector<hsize_t> boundaryDims = {3};
    auto space = H5Screate_simple(int_c(boundaryDims.size()), boundaryDims.data(), nullptr);
    auto att =
        H5Acreate(config_.boxGroupId, "boundary", boundaryType, space, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_HDF5(H5Awrite(att, boundaryType, "periodicperiodicperiodic"));
    CHECK_HDF5(H5Aclose(att));
    CHECK_HDF5(H5Sclose(space));
    CHECK_HDF5(H5Tclose(boundaryType));

    config_.edgesGroupId = createGroup(config_.boxGroupId, "edges");
    config_.edgesStepSetId = createChunkedDataset(
        config_.edgesGroupId, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    config_.edgesTimeSetId = createChunkedDataset(
        config_.edgesGroupId, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    config_.edgesValueSetId = createChunkedDataset(
        config_.edgesGroupId, std::vector<hsize_t>{1, 3}, "value", H5T_NATIVE_DOUBLE);
}

hid_t DumpH5MDParallelImpl::createChunkedDataset(const hid_t& groupId,
                                                 const std::vector<hsize_t>& dims,
                                                 const std::string& name,
                                                 const hid_t& dtype) const
{
    std::vector<hsize_t> max_dims = dims;
    max_dims[0] = H5S_UNLIMITED;
    hid_t fileSpace = H5Screate_simple(dims.size(), dims.data(), max_dims.data());

    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(plist, H5D_CHUNKED);

    const std::vector<hsize_t> chunk_dims = dims;
    H5Pset_chunk(plist, dims.size(), chunk_dims.data());

    auto datasetId =
        H5Dcreate(groupId, name.c_str(), dtype, fileSpace, H5P_DEFAULT, plist, H5P_DEFAULT);

    H5Pclose(plist);
    H5Sclose(fileSpace);

    return datasetId;
}

void DumpH5MDParallelImpl::closeDataset(const hid_t& datasetId) const { H5Dclose(datasetId); }

void DumpH5MDParallelImpl::dumpStep(const data::Subdomain& subdomain,
                                    const data::Atoms& atoms,
                                    const idx_t step,
                                    const real_t dt)
{
    data::HostAtoms h_atoms(atoms);  // NOLINT

    updateCache(h_atoms);

    appendEdges(step, dt, subdomain);
    appendCharges(step, dt, h_atoms);
    appendForces(step, dt, h_atoms);
    appendMasses(step, dt, h_atoms);
    appendPositions(step, dt, h_atoms);
    appendRelativeMasses(step, dt, h_atoms);
    appendTypes(step, dt, h_atoms);
    appendVelocities(step, dt, h_atoms);
    config_.saveCount += 1;
}

void DumpH5MDParallelImpl::appendEdges(const idx_t& step,
                                       const real_t& dt,
                                       const data::Subdomain& subdomain) const
{
    appendData(config_.edgesStepSetId, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(
        config_.edgesTimeSetId, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    appendData(
        config_.edgesValueSetId,
        std::vector<real_t>{subdomain.diameter[0], subdomain.diameter[1], subdomain.diameter[2]},
        std::vector<hsize_t>{1, 3});
}

void DumpH5MDParallelImpl::appendCharges(const idx_t& step,
                                         const real_t& dt,
                                         const data::HostAtoms& atoms) const
{
    appendData(config_.chargesStepSetId, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(
        config_.chargesTimeSetId, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    hsize_t numberLocalAtoms = atoms.numLocalAtoms;
    constexpr int64_t dimensions = 1;
    std::vector<real_t> charges;
    charges.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < numLocalParticles; ++idx)
    {
        charges.emplace_back(atoms.getCharge()(idx));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(charges.size()), numLocalParticles * dimensions);
    appendDataParallel(
        config_.chargesValueSetId, charges, std::vector<hsize_t>{1, numberLocalAtoms, dimensions});
}

void DumpH5MDParallelImpl::appendForces(const idx_t& step,
                                        const real_t& dt,
                                        const data::HostAtoms& atoms) const
{
    appendData(config_.forceStepSetId, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(
        config_.forceTimeSetId, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    hsize_t numberLocalAtoms = atoms.numLocalAtoms;
    constexpr int64_t dimensions = 3;
    std::vector<real_t> positions;
    positions.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        positions.emplace_back(atoms.getForce()(idx, 0));
        positions.emplace_back(atoms.getForce()(idx, 1));
        positions.emplace_back(atoms.getForce()(idx, 2));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(positions.size()), numLocalParticles * dimensions);
    appendDataParallel(
        config_.forceValueSetId, positions, std::vector<hsize_t>{1, numberLocalAtoms, dimensions});
}

void DumpH5MDParallelImpl::appendMasses(const idx_t& step,
                                        const real_t& dt,
                                        const data::HostAtoms& atoms) const
{
    appendData(config_.massStepSetId, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(
        config_.massTimeSetId, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    hsize_t numberLocalAtoms = atoms.numLocalAtoms;
    constexpr int64_t dimensions = 1;
    std::vector<real_t> masses;
    masses.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < numLocalParticles; ++idx)
    {
        masses.emplace_back(atoms.getMass()(idx));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(masses.size()), numLocalParticles * dimensions);
    appendDataParallel(
        config_.massValueSetId, masses, std::vector<hsize_t>{1, numberLocalAtoms, dimensions});
}

void DumpH5MDParallelImpl::appendPositions(const idx_t& step,
                                           const real_t& dt,
                                           const data::HostAtoms& atoms) const
{
    appendData(config_.posStepSetId, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(
        config_.posTimeSetId, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    hsize_t numberLocalAtoms = atoms.numLocalAtoms;
    constexpr int64_t dimensions = 3;
    std::vector<real_t> positions;
    positions.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        positions.emplace_back(atoms.getPos()(idx, 0));
        positions.emplace_back(atoms.getPos()(idx, 1));
        positions.emplace_back(atoms.getPos()(idx, 2));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(positions.size()), numLocalParticles * dimensions);
    appendDataParallel(
        config_.posValueSetId, positions, std::vector<hsize_t>{1, numberLocalAtoms, dimensions});
}

void DumpH5MDParallelImpl::appendRelativeMasses(const idx_t& step,
                                                const real_t& dt,
                                                const data::HostAtoms& atoms) const
{
    appendData(config_.relativeMassStepSetId, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(config_.relativeMassTimeSetId,
               std::vector<real_t>{real_c(step) * dt},
               std::vector<hsize_t>{1});
    hsize_t numberLocalAtoms = atoms.numLocalAtoms;
    constexpr int64_t dimensions = 1;
    std::vector<real_t> relativeMasses;
    relativeMasses.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < numLocalParticles; ++idx)
    {
        relativeMasses.emplace_back(atoms.getRelativeMass()(idx));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(relativeMasses.size()), numLocalParticles * dimensions);
    appendDataParallel(config_.relativeMassValueSetId,
                       relativeMasses,
                       std::vector<hsize_t>{1, numberLocalAtoms, dimensions});
}

void DumpH5MDParallelImpl::appendTypes(const idx_t& step,
                                       const real_t& dt,
                                       const data::HostAtoms& atoms) const
{
    appendData(config_.typeStepSetId, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(
        config_.typeTimeSetId, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    hsize_t numberLocalAtoms = atoms.numLocalAtoms;
    constexpr int64_t dimensions = 1;
    std::vector<real_t> types;
    types.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < numLocalParticles; ++idx)
    {
        types.emplace_back(atoms.getType()(idx));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(types.size()), numLocalParticles * dimensions);
    appendDataParallel(
        config_.typeValueSetId, types, std::vector<hsize_t>{1, numberLocalAtoms, dimensions});
}

void DumpH5MDParallelImpl::appendVelocities(const idx_t& step,
                                            const real_t& dt,
                                            const data::HostAtoms& atoms) const
{
    appendData(config_.velStepSetId, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(
        config_.velTimeSetId, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    hsize_t numberLocalAtoms = atoms.numLocalAtoms;
    constexpr int64_t dimensions = 3;
    std::vector<real_t> velocities;
    velocities.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        velocities.emplace_back(atoms.getVel()(idx, 0));
        velocities.emplace_back(atoms.getVel()(idx, 1));
        velocities.emplace_back(atoms.getVel()(idx, 2));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(velocities.size()), numLocalParticles * dimensions);
    appendDataParallel(
        config_.velValueSetId, velocities, std::vector<hsize_t>{1, numberLocalAtoms, dimensions});
}

template <typename T>
void DumpH5MDParallelImpl::appendDataParallel(const hid_t datasetId,
                                              const std::vector<T>& data,
                                              const std::vector<hsize_t>& dims) const
{
    std::vector<hsize_t> newSize = dims;
    newSize[0] = config_.saveCount + 1;
    H5Dset_extent(datasetId, newSize.data());

    const auto fileSpace = H5Dget_space(datasetId);

    std::vector<hsize_t> offset(dims.size(), 0);
    offset[0] = config_.saveCount;
    offset[1] = particleOffset;
    std::vector<hsize_t> stride(dims.size(), 1);
    std::vector<hsize_t> count(dims.size(), 1);

    CHECK_HDF5(H5Sselect_hyperslab(
        fileSpace, H5S_SELECT_SET, offset.data(), stride.data(), count.data(), dims.data()));

    std::vector<hsize_t> localOffset(dims.size(), 0);
    const hid_t memorySpace = H5Screate_simple(dims.size(), dims.data(), NULL);
    CHECK_HDF5(H5Sselect_hyperslab(
        memorySpace, H5S_SELECT_SET, localOffset.data(), stride.data(), count.data(), dims.data()));

    auto propertyList = CHECK_HDF5(H5Pcreate(H5P_DATASET_XFER));
    CHECK_HDF5(H5Pset_dxpl_mpio(propertyList, H5FD_MPIO_COLLECTIVE));
    CHECK_HDF5(
        H5Dwrite(datasetId, typeToHDF5<T>(), memorySpace, fileSpace, propertyList, data.data()));

    CHECK_HDF5(H5Pclose(propertyList));
    H5Sclose(fileSpace);
    H5Sclose(memorySpace);
}

template <typename T>
void DumpH5MDParallelImpl::appendData(const hid_t datasetId,
                                      const std::vector<T>& data,
                                      const std::vector<hsize_t>& dims) const
{
    const hid_t memorySpace = H5Screate_simple(dims.size(), dims.data(), NULL);

    std::vector<hsize_t> newSize = dims;
    newSize[0] = config_.saveCount + 1;
    H5Dset_extent(datasetId, newSize.data());

    const auto fileSpace = H5Dget_space(datasetId);

    std::vector<hsize_t> start(dims.size(), 0);
    start[0] = config_.saveCount;
    std::vector<hsize_t> count = dims;
    count[0] = 1;
    H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);

    H5Dwrite(datasetId, typeToHDF5<T>(), memorySpace, fileSpace, H5P_DEFAULT, data.data());

    H5Sclose(fileSpace);
    H5Sclose(memorySpace);
}

template <typename T>
void DumpH5MDParallelImpl::writeParallel(hid_t fileId,
                                         const std::string& name,
                                         const std::vector<hsize_t>& globalDims,
                                         const std::vector<hsize_t>& localDims,
                                         const std::vector<T>& data)
{
    MRMD_HOST_CHECK_EQUAL(globalDims.size(), localDims.size());
    MRMD_HOST_CHECK_EQUAL(
        data.size(),
        std::accumulate(localDims.begin(), localDims.end(), hsize_t(1), std::multiplies<>()));

    auto dataspace =
        CHECK_HDF5(H5Screate_simple(int_c(globalDims.size()), globalDims.data(), nullptr));
    auto dataset = CHECK_HDF5(H5Dcreate(
        fileId, name.c_str(), typeToHDF5<T>(), dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    std::vector<hsize_t> offset(globalDims.size(), 0);
    offset[1] = particleOffset;
    std::vector<hsize_t> stride(globalDims.size(), 1);
    std::vector<hsize_t> count(globalDims.size(), 1);
    for (auto i = 0; i < int_c(globalDims.size()); ++i)
    {
        MRMD_HOST_CHECK_LESSEQUAL(
            localDims[i] + offset[i], globalDims[i], fmt::format("i = {}", i));
    }
    auto dstSpace = CHECK_HDF5(H5Dget_space(dataset));
    CHECK_HDF5(H5Sselect_hyperslab(
        dstSpace, H5S_SELECT_SET, offset.data(), stride.data(), count.data(), localDims.data()));

    std::vector<hsize_t> localOffset(globalDims.size(), 0);
    auto srcSpace =
        CHECK_HDF5(H5Screate_simple(int_c(localDims.size()), localDims.data(), nullptr));
    CHECK_HDF5(H5Sselect_hyperslab(srcSpace,
                                   H5S_SELECT_SET,
                                   localOffset.data(),
                                   stride.data(),
                                   count.data(),
                                   localDims.data()));

    auto datawrite = CHECK_HDF5(H5Pcreate(H5P_DATASET_XFER));
    CHECK_HDF5(H5Pset_dxpl_mpio(datawrite, H5FD_MPIO_COLLECTIVE));
    CHECK_HDF5(H5Dwrite(dataset, typeToHDF5<T>(), srcSpace, dstSpace, datawrite, data.data()));

    CHECK_HDF5(H5Pclose(datawrite));
    CHECK_HDF5(H5Sclose(dstSpace));
    CHECK_HDF5(H5Sclose(srcSpace));
    CHECK_HDF5(H5Dclose(dataset));
    CHECK_HDF5(H5Sclose(dataspace));
}

void DumpH5MDParallelImpl::writeHeader(hid_t fileId) const
{
    auto group1 = CHECK_HDF5(H5Gcreate(fileId, "/h5md", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    auto group2 =
        CHECK_HDF5(H5Gcreate(fileId, "/h5md/author", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    auto group3 =
        CHECK_HDF5(H5Gcreate(fileId, "/h5md/creator", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    CHECK_HDF5(H5Gclose(group1));
    CHECK_HDF5(H5Gclose(group2));
    CHECK_HDF5(H5Gclose(group3));

    std::vector<int> data = {1, 1};
    CHECK_HDF5(H5LTset_attribute_int(fileId, "/h5md", "version", data.data(), data.size()));

    CHECK_HDF5(H5LTset_attribute_string(fileId, "/h5md/author", "name", config_.author.c_str()));

    CHECK_HDF5(H5LTset_attribute_string(fileId, "/h5md/creator", "name", PROJECT_NAME.c_str()));
    CHECK_HDF5(H5LTset_attribute_string(fileId, "/h5md/creator", "version", MRMD_VERSION.c_str()));
}

void DumpH5MDParallelImpl::writeBox(hid_t fileId, const data::Subdomain& subdomain) const
{
    std::string groupName = "/particles/" + config_.particleSubGroupName + "/box";
    auto group =
        CHECK_HDF5(H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    std::vector<int> dims = {3};
    CHECK_HDF5(
        H5LTset_attribute_int(fileId, groupName.c_str(), "dimension", dims.data(), dims.size()));

    auto boundaryType = H5Tcopy(H5T_C_S1);
    CHECK_HDF5(H5Tset_size(boundaryType, 8));
    CHECK_HDF5(H5Tset_strpad(boundaryType, H5T_STR_NULLPAD));
    std::vector<hsize_t> boundaryDims = {3};
    auto space = H5Screate_simple(int_c(boundaryDims.size()), boundaryDims.data(), nullptr);
    auto att = H5Acreate(group, "boundary", boundaryType, space, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_HDF5(H5Awrite(att, boundaryType, "periodicperiodicperiodic"));
    CHECK_HDF5(H5Aclose(att));
    CHECK_HDF5(H5Sclose(space));
    CHECK_HDF5(H5Tclose(boundaryType));

    std::string edgesGroupName = groupName + "/edges";
    auto edgesGroup =
        H5Gcreate(fileId, edgesGroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<hsize_t> edgesStepDims = {1};
    std::vector<int64_t> step = {0};
    std::string edgesStepDataset = edgesGroupName + "/step";
    CHECK_HDF5(H5LTmake_dataset(fileId,
                                edgesStepDataset.c_str(),
                                1,
                                edgesStepDims.data(),
                                typeToHDF5<int64_t>(),
                                step.data()));
    std::vector<hsize_t> edgesValueDims = {1, 3};
    std::string edgesValueDataset = edgesGroupName + "/value";
    CHECK_HDF5(H5LTmake_dataset(fileId,
                                edgesValueDataset.c_str(),
                                2,
                                edgesValueDims.data(),
                                typeToHDF5<double>(),
                                subdomain.diameter.data()));
    CHECK_HDF5(H5Gclose(edgesGroup));

    CHECK_HDF5(H5LTset_attribute_double(fileId,
                                        groupName.c_str(),
                                        "minCorner",
                                        subdomain.minCorner.data(),
                                        subdomain.minCorner.size()));
    CHECK_HDF5(H5LTset_attribute_double(fileId,
                                        groupName.c_str(),
                                        "maxCorner",
                                        subdomain.maxCorner.data(),
                                        subdomain.maxCorner.size()));
    CHECK_HDF5(H5LTset_attribute_double(fileId,
                                        groupName.c_str(),
                                        "ghostLayerThickness",
                                        subdomain.ghostLayerThickness.data(),
                                        subdomain.ghostLayerThickness.size()));

    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallelImpl::writePos(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 3;  ///< dimensions of the property

    std::string groupName = "/particles/" + config_.particleSubGroupName + "/" + config_.posDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getPos()(idx, 0));
        data.emplace_back(atoms.getPos()(idx, 1));
        data.emplace_back(atoms.getPos()(idx, 2));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

    std::vector<hsize_t> localDims = {1, uint64_c(numLocalParticles), dimensions};
    std::vector<hsize_t> globalDims = {1, uint64_c(numTotalParticles), dimensions};

    std::string dataset_name = groupName + "/value";
    writeParallel(fileId, dataset_name, globalDims, localDims, data);

    std::vector<hsize_t> dims = {1};
    std::vector<int64_t> step = {0};
    std::vector<double> time = {0};
    std::string stepDataset = groupName + "/step";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, stepDataset.c_str(), 1, dims.data(), typeToHDF5<int64_t>(), step.data()));
    std::string timeDataset = groupName + "/time";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, timeDataset.c_str(), 1, dims.data(), typeToHDF5<double>(), time.data()));
    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallelImpl::writeVel(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 3;  ///< dimensions of the property

    std::string groupName = "/particles/" + config_.particleSubGroupName + "/" + config_.velDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getVel()(idx, 0));
        data.emplace_back(atoms.getVel()(idx, 1));
        data.emplace_back(atoms.getVel()(idx, 2));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

    std::vector<hsize_t> localDims = {1, uint64_c(numLocalParticles), dimensions};
    std::vector<hsize_t> globalDims = {1, uint64_c(numTotalParticles), dimensions};

    std::string dataset_name = groupName + "/value";
    writeParallel(fileId, dataset_name, globalDims, localDims, data);

    std::vector<hsize_t> dims = {1};
    std::vector<int64_t> step = {0};
    std::vector<double> time = {0};
    std::string stepDataset = groupName + "/step";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, stepDataset.c_str(), 1, dims.data(), typeToHDF5<int64_t>(), step.data()));
    std::string timeDataset = groupName + "/time";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, timeDataset.c_str(), 1, dims.data(), typeToHDF5<double>(), time.data()));
    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallelImpl::writeForce(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 3;  ///< dimensions of the property

    std::string groupName =
        "/particles/" + config_.particleSubGroupName + "/" + config_.forceDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getForce()(idx, 0));
        data.emplace_back(atoms.getForce()(idx, 1));
        data.emplace_back(atoms.getForce()(idx, 2));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

    std::vector<hsize_t> localDims = {1, uint64_c(numLocalParticles), dimensions};
    std::vector<hsize_t> globalDims = {1, uint64_c(numTotalParticles), dimensions};

    std::string dataset_name = groupName + "/value";
    writeParallel(fileId, dataset_name, globalDims, localDims, data);

    std::vector<hsize_t> dims = {1};
    std::vector<int64_t> step = {0};
    std::vector<double> time = {0};
    std::string stepDataset = groupName + "/step";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, stepDataset.c_str(), 1, dims.data(), typeToHDF5<int64_t>(), step.data()));
    std::string timeDataset = groupName + "/time";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, timeDataset.c_str(), 1, dims.data(), typeToHDF5<double>(), time.data()));
    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallelImpl::writeType(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = idx_t;
    constexpr int64_t dimensions = 1;  ///< dimensions of the property

    std::string groupName =
        "/particles/" + config_.particleSubGroupName + "/" + config_.typeDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getType()(idx));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

    std::vector<hsize_t> localDims = {1, uint64_c(numLocalParticles), dimensions};
    std::vector<hsize_t> globalDims = {1, uint64_c(numTotalParticles), dimensions};

    std::string dataset_name = groupName + "/value";
    writeParallel(fileId, dataset_name, globalDims, localDims, data);

    std::vector<hsize_t> dims = {1};
    std::vector<int64_t> step = {0};
    std::vector<double> time = {0};
    std::string stepDataset = groupName + "/step";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, stepDataset.c_str(), 1, dims.data(), typeToHDF5<int64_t>(), step.data()));
    std::string timeDataset = groupName + "/time";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, timeDataset.c_str(), 1, dims.data(), typeToHDF5<double>(), time.data()));
    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallelImpl::writeMass(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 1;  ///< dimensions of the property

    std::string groupName =
        "/particles/" + config_.particleSubGroupName + "/" + config_.massDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getMass()(idx));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

    std::vector<hsize_t> localDims = {1, uint64_c(numLocalParticles), dimensions};
    std::vector<hsize_t> globalDims = {1, uint64_c(numTotalParticles), dimensions};

    std::string dataset_name = groupName + "/value";
    writeParallel(fileId, dataset_name, globalDims, localDims, data);

    std::vector<hsize_t> dims = {1};
    std::vector<int64_t> step = {0};
    std::vector<double> time = {0};
    std::string stepDataset = groupName + "/step";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, stepDataset.c_str(), 1, dims.data(), typeToHDF5<int64_t>(), step.data()));
    std::string timeDataset = groupName + "/time";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, timeDataset.c_str(), 1, dims.data(), typeToHDF5<double>(), time.data()));
    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallelImpl::writeCharge(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 1;  ///< dimensions of the property

    std::string groupName =
        "/particles/" + config_.particleSubGroupName + "/" + config_.chargeDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getCharge()(idx));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

    std::vector<hsize_t> localDims = {1, uint64_c(numLocalParticles), dimensions};
    std::vector<hsize_t> globalDims = {1, uint64_c(numTotalParticles), dimensions};

    std::string dataset_name = groupName + "/value";
    writeParallel(fileId, dataset_name, globalDims, localDims, data);

    std::vector<hsize_t> dims = {1};
    std::vector<int64_t> step = {0};
    std::vector<double> time = {0};
    std::string stepDataset = groupName + "/step";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, stepDataset.c_str(), 1, dims.data(), typeToHDF5<int64_t>(), step.data()));
    std::string timeDataset = groupName + "/time";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, timeDataset.c_str(), 1, dims.data(), typeToHDF5<double>(), time.data()));
    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallelImpl::writeRelativeMass(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 1;  ///< dimensions of the property

    std::string groupName =
        "/particles/" + config_.particleSubGroupName + "/" + config_.relativeMassDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getRelativeMass()(idx));
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

    std::vector<hsize_t> localDims = {1, uint64_c(numLocalParticles), dimensions};
    std::vector<hsize_t> globalDims = {1, uint64_c(numTotalParticles), dimensions};

    std::string dataset_name = groupName + "/value";
    writeParallel(fileId, dataset_name, globalDims, localDims, data);

    std::vector<hsize_t> dims = {1};
    std::vector<int64_t> step = {0};
    std::vector<double> time = {0};
    std::string stepDataset = groupName + "/step";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, stepDataset.c_str(), 1, dims.data(), typeToHDF5<int64_t>(), step.data()));
    std::string timeDataset = groupName + "/time";
    CHECK_HDF5(H5LTmake_dataset(
        fileId, timeDataset.c_str(), 1, dims.data(), typeToHDF5<double>(), time.data()));
    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallelImpl::updateCache(const data::HostAtoms& atoms)
{
    numLocalParticles = atoms.numLocalAtoms;
    MPI_Allreduce(reinterpret_cast<const void*>(&numLocalParticles),
                  reinterpret_cast<void*>(&numTotalParticles),
                  1,
                  MPI_INT64_T,
                  MPI_SUM,
                  config_.mpiInfo->comm);

    MPI_Exscan(&numLocalParticles, &particleOffset, 1, MPI_INT64_T, MPI_SUM, config_.mpiInfo->comm);
    if (config_.mpiInfo->rank == 0) particleOffset = 0;
}

void DumpH5MDParallelImpl::dump(const std::string& filename,
                                const data::Subdomain& subdomain,
                                const data::Atoms& atoms)
{
    data::HostAtoms h_atoms(atoms);  // NOLINT

    updateCache(h_atoms);

    MPI_Info info = MPI_INFO_NULL;

    auto plist = CHECK_HDF5(H5Pcreate(H5P_FILE_ACCESS));
    CHECK_HDF5(H5Pset_fapl_mpio(plist, config_.mpiInfo->comm, info));

    auto file_id = CHECK_HDF5(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist));

    auto group1 =
        CHECK_HDF5(H5Gcreate(file_id, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    std::string particleSubGroup = "/particles/" + config_.particleSubGroupName;
    auto group2 = CHECK_HDF5(
        H5Gcreate(file_id, particleSubGroup.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    writeHeader(file_id);
    writeBox(file_id, subdomain);
    if (config_.dumpPos) writePos(file_id, h_atoms);
    if (config_.dumpVel) writeVel(file_id, h_atoms);
    if (config_.dumpForce) writeForce(file_id, h_atoms);
    if (config_.dumpType) writeType(file_id, h_atoms);
    if (config_.dumpMass) writeMass(file_id, h_atoms);
    if (config_.dumpCharge) writeCharge(file_id, h_atoms);
    if (config_.dumpRelativeMass) writeRelativeMass(file_id, h_atoms);

    CHECK_HDF5(H5Gclose(group1));
    CHECK_HDF5(H5Gclose(group2));

    CHECK_HDF5(H5Fclose(file_id));
}
}  // namespace impl

void DumpH5MDParallel::open(const std::string& filename, const data::Atoms& atoms)
{
    impl::DumpH5MDParallelImpl helper(*this);
    helper.open(filename, atoms);
}

void DumpH5MDParallel::dumpStep(const data::Subdomain& subdomain,
                                const data::Atoms& atoms,
                                const idx_t& step,
                                const real_t& dt)
{
    impl::DumpH5MDParallelImpl helper(*this);
    helper.dumpStep(subdomain, atoms, step, dt);
}

void DumpH5MDParallel::close()
{
    impl::DumpH5MDParallelImpl helper(*this);
    helper.close();
}

void DumpH5MDParallel::dump(const std::string& filename,
                            const data::Subdomain& subdomain,
                            const data::Atoms& atoms)
{
    impl::DumpH5MDParallelImpl helper(*this);
    helper.dump(filename, subdomain, atoms);
}
#else
void DumpH5MDParallel::open(const std::string& /*filename*/, const data::Atoms& /*atoms*/)
{
    MRMD_HOST_CHECK(false, "HDF5 Support not available!");
    exit(EXIT_FAILURE);
}

void DumpH5MDParallel::close();
{
    MRMD_HOST_CHECK(false, "HDF5 Support not available!");
    exit(EXIT_FAILURE);
}
void DumpH5MDParallel::dumpStep(const data::Subdomain& /*subdomain*/,
                                const data::Atoms& /*atoms*/,
                                const idx_t& /*step*/,
                                const real_t& /*dt*/)
{
    MRMD_HOST_CHECK(false, "HDF5 Support not available!");
    exit(EXIT_FAILURE);
}

void DumpH5MDParallel::dump(const std::string& /*filename*/,
                            const data::Subdomain& /*subdomain*/,
                            const data::Atoms& /*atoms*/)
{
    MRMD_HOST_CHECK(false, "HDF5 Support not available!");
    exit(EXIT_FAILURE);
}
#endif

}  // namespace mrmd::io