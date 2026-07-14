// Copyright 2024 Sebastian Eibl
// Copyright 2026 Julian Friedrich Hille
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

#include "DumpH5MD.hpp"

#include <algorithm>
#include <functional>
#include <numeric>

#include "assert/assert.hpp"
#ifdef MRMD_ENABLE_HDF5
#include "hdf5.hpp"
#endif
#include "version.hpp"

namespace mrmd::io
{
DumpH5MD::DumpH5MD(const std::string& authorArg, const std::string& particleGroupNameArg)
    : author(authorArg), particleGroupName(particleGroupNameArg)
{
}

#ifdef MRMD_ENABLE_HDF5
namespace impl
{
/**
 * This class collects everything needed for the HDF5 dump. If HDF5 is disabled,
 * this class can be skipped and all HDF5 dependencies are gone.
 */
class DumpH5MDImpl
{
public:
    explicit DumpH5MDImpl(DumpH5MD& config) : config_(config), state_(config.state_) {}

    void open(const std::string& filename,
              const data::Subdomain& subdomain,
              const data::Atoms& atoms);

    void dumpStep(const data::Subdomain& subdomain,
                  const data::Atoms& atoms,
                  const idx_t step,
                  const real_t dt);

    void close();

    void dump(const std::string& filename,
              const data::Subdomain& subdomain,
              const data::Atoms& atoms);

private:
    hid_t createFile(const std::string& filename) const;
    void closeFile(hid_t& fileId) const;
    hid_t createGroup(const hid_t& parentElementId, const std::string& groupName) const;
    void closeGroup(hid_t& groupId) const;
    void openBox(const data::Subdomain& subdomain) const;
    hid_t createChunkedDataset(const hid_t& groupId,
                               const std::vector<hsize_t>& dims,
                               const std::string& name,
                               const hid_t& dtype) const;
    void closeDataset(hid_t& datasetId) const;

    template <typename T>
    void appendData(const hid_t datasetId,
                    const std::vector<T>& data,
                    const std::vector<hsize_t>& dims) const;

    void appendEdges(const idx_t& step, const real_t& dt, const data::Subdomain& subdomain) const;

    void openParticleElement(const std::string& datasetName,
                             const std::vector<hsize_t>& valueDims,
                             const hid_t& valueType,
                             DumpH5MD::ElementHandles& handles) const;
    void closeParticleElement(DumpH5MD::ElementHandles& handles) const;

    template <typename T, typename Extractor>
    void writeParticleElement(hid_t fileId,
                              const std::string& datasetName,
                              const data::HostAtoms& atoms,
                              const int64_t dimensions,
                              Extractor&& extractor);

    template <typename T, typename Extractor>
    void appendParticleElement(const idx_t& step,
                               const real_t& dt,
                               const data::HostAtoms& atoms,
                               const int64_t dimensions,
                               Extractor&& extractor,
                               const DumpH5MD::ElementHandles& handles) const;

    template <typename T, typename Extractor>
    std::vector<T> collectPropertyData(const data::HostAtoms& atoms,
                                       const int64_t dimensions,
                                       Extractor&& extractor) const;

    void updateCache(const data::HostAtoms& atoms);

    void writeHeader(hid_t fileId) const;
    void writeBox(hid_t fileId, const data::Subdomain& subdomain) const;
    template <typename T>
    void write(hid_t fileId,
               const std::string& name,
               const std::vector<hsize_t>& dims,
               const std::vector<T>& data);

    DumpH5MD& config_;
    DumpH5MD::State& state_;

    int64_t numLocalParticles = -1;
};

template <typename T>
void DumpH5MDImpl::write(hid_t fileId,
                         const std::string& name,
                         const std::vector<hsize_t>& dims,
                         const std::vector<T>& data)
{
    MRMD_HOST_CHECK_EQUAL(
        data.size(), std::accumulate(dims.begin(), dims.end(), hsize_t(1), std::multiplies<>()));

    auto dataspace = CHECK_HDF5(H5Screate_simple(int_c(dims.size()), dims.data(), nullptr));
    auto dataset = CHECK_HDF5(H5Dcreate(
        fileId, name.c_str(), typeToHDF5<T>(), dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    CHECK_HDF5(H5Dwrite(dataset, typeToHDF5<T>(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()));

    CHECK_HDF5(H5Dclose(dataset));
    CHECK_HDF5(H5Sclose(dataspace));
}

template <typename T, typename Extractor>
std::vector<T> DumpH5MDImpl::collectPropertyData(const data::HostAtoms& atoms,
                                                 const int64_t dimensions,
                                                 Extractor&& extractor) const
{
    std::vector<T> values;
    values.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        for (int64_t dim = 0; dim < dimensions; ++dim)
        {
            values.emplace_back(extractor(idx, dim));
        }
    }
    MRMD_HOST_CHECK_EQUAL(int64_c(values.size()), numLocalParticles * dimensions);
    return values;
}

template <typename T, typename Extractor>
void DumpH5MDImpl::writeParticleElement(hid_t fileId,
                                        const std::string& datasetName,
                                        const data::HostAtoms& atoms,
                                        const int64_t dimensions,
                                        Extractor&& extractor)
{
    const std::string groupName = "/particles/" + config_.particleGroupName + "/" + datasetName;
    const hid_t group =
        CHECK_HDF5(H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    const auto values =
        collectPropertyData<T>(atoms, dimensions, std::forward<Extractor>(extractor));
    const std::vector<hsize_t> valueDims = {1, uint64_c(numLocalParticles), uint64_c(dimensions)};
    write(fileId, groupName + "/value", valueDims, values);

    std::vector<hsize_t> dims = {1};
    std::vector<int64_t> step = {0};
    std::vector<double> time = {0};
    CHECK_HDF5(H5LTmake_dataset(
        fileId, (groupName + "/step").c_str(), 1, dims.data(), typeToHDF5<int64_t>(), step.data()));
    CHECK_HDF5(H5LTmake_dataset(
        fileId, (groupName + "/time").c_str(), 1, dims.data(), typeToHDF5<double>(), time.data()));
    CHECK_HDF5(H5Gclose(group));
}

template <typename T, typename Extractor>
void DumpH5MDImpl::appendParticleElement(const idx_t& step,
                                         const real_t& dt,
                                         const data::HostAtoms& atoms,
                                         const int64_t dimensions,
                                         Extractor&& extractor,
                                         const DumpH5MD::ElementHandles& handles) const
{
    const auto values =
        collectPropertyData<T>(atoms, dimensions, std::forward<Extractor>(extractor));
    const std::vector<hsize_t> valueDims = {1, uint64_c(atoms.numLocalAtoms), uint64_c(dimensions)};
    appendData(handles.step, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(handles.time, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    appendData(handles.value, values, valueDims);
}

void DumpH5MDImpl::writeHeader(hid_t fileId) const
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

void DumpH5MDImpl::writeBox(hid_t fileId, const data::Subdomain& subdomain) const
{
    std::string groupName = "/particles/" + config_.particleGroupName + "/box";
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

hid_t DumpH5MDImpl::createFile(const std::string& filename) const
{
    auto fileId = CHECK_HDF5(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    return fileId;
}

void DumpH5MDImpl::closeFile(hid_t& fileId) const
{
    if (fileId < 0) return;
    CHECK_HDF5(H5Fclose(fileId));
    fileId = -1;
}

hid_t DumpH5MDImpl::createGroup(const hid_t& parentElementId, const std::string& groupName) const
{
    auto groupId = CHECK_HDF5(
        H5Gcreate(parentElementId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    return groupId;
}

void DumpH5MDImpl::closeGroup(hid_t& groupId) const
{
    if (groupId < 0) return;
    CHECK_HDF5(H5Gclose(groupId));
    groupId = -1;
}

void DumpH5MDImpl::closeDataset(hid_t& datasetId) const
{
    if (datasetId < 0) return;
    CHECK_HDF5(H5Dclose(datasetId));
    datasetId = -1;
}

void DumpH5MDImpl::openParticleElement(const std::string& datasetName,
                                       const std::vector<hsize_t>& valueDims,
                                       const hid_t& valueType,
                                       DumpH5MD::ElementHandles& handles) const
{
    handles.group = createGroup(state_.particleSubGroupId, datasetName);
    handles.step =
        createChunkedDataset(handles.group, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    handles.time =
        createChunkedDataset(handles.group, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    handles.value = createChunkedDataset(handles.group, valueDims, "value", valueType);
}

void DumpH5MDImpl::closeParticleElement(DumpH5MD::ElementHandles& handles) const
{
    closeDataset(handles.value);
    closeDataset(handles.time);
    closeDataset(handles.step);
    closeGroup(handles.group);
}

hid_t DumpH5MDImpl::createChunkedDataset(const hid_t& groupId,
                                         const std::vector<hsize_t>& dims,
                                         const std::string& name,
                                         const hid_t& dtype) const
{
    std::vector<hsize_t> max_dims = dims;
    max_dims[0] = H5S_UNLIMITED;
    if (dims.size() == 3)
    {
        max_dims[1] = H5S_UNLIMITED;
    }

    const hid_t fileSpace =
        CHECK_HDF5(H5Screate_simple(int_c(dims.size()), dims.data(), max_dims.data()));

    const hid_t plist = CHECK_HDF5(H5Pcreate(H5P_DATASET_CREATE));
    CHECK_HDF5(H5Pset_layout(plist, H5D_CHUNKED));
    CHECK_HDF5(H5Pset_chunk(plist, int_c(dims.size()), dims.data()));

    const hid_t datasetId = CHECK_HDF5(
        H5Dcreate(groupId, name.c_str(), dtype, fileSpace, H5P_DEFAULT, plist, H5P_DEFAULT));

    CHECK_HDF5(H5Pclose(plist));
    CHECK_HDF5(H5Sclose(fileSpace));

    return datasetId;
}

void DumpH5MDImpl::openBox(const data::Subdomain& subdomain) const
{
    state_.boxGroupId = createGroup(state_.particleSubGroupId, "box");

    std::vector<int> dims = {3};
    CHECK_HDF5(H5LTset_attribute_int(
        state_.particleSubGroupId, "box", "dimension", dims.data(), dims.size()));
    CHECK_HDF5(H5LTset_attribute_double(state_.particleSubGroupId,
                                        "box",
                                        "minCorner",
                                        subdomain.minCorner.data(),
                                        subdomain.minCorner.size()));
    CHECK_HDF5(H5LTset_attribute_double(state_.particleSubGroupId,
                                        "box",
                                        "maxCorner",
                                        subdomain.maxCorner.data(),
                                        subdomain.maxCorner.size()));
    CHECK_HDF5(H5LTset_attribute_double(state_.particleSubGroupId,
                                        "box",
                                        "ghostLayerThickness",
                                        subdomain.ghostLayerThickness.data(),
                                        subdomain.ghostLayerThickness.size()));

    auto boundaryType = H5Tcopy(H5T_C_S1);
    CHECK_HDF5(H5Tset_size(boundaryType, 8));
    CHECK_HDF5(H5Tset_strpad(boundaryType, H5T_STR_NULLPAD));
    std::vector<hsize_t> boundaryDims = {3};
    auto space = H5Screate_simple(int_c(boundaryDims.size()), boundaryDims.data(), nullptr);
    auto att =
        H5Acreate(state_.boxGroupId, "boundary", boundaryType, space, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_HDF5(H5Awrite(att, boundaryType, "periodicperiodicperiodic"));
    CHECK_HDF5(H5Aclose(att));
    CHECK_HDF5(H5Sclose(space));
    CHECK_HDF5(H5Tclose(boundaryType));

    state_.edges.group = createGroup(state_.boxGroupId, "edges");
    state_.edges.step =
        createChunkedDataset(state_.edges.group, std::vector<hsize_t>{1}, "step", H5T_NATIVE_INT64);
    state_.edges.time = createChunkedDataset(
        state_.edges.group, std::vector<hsize_t>{1}, "time", H5T_NATIVE_DOUBLE);
    state_.edges.value = createChunkedDataset(
        state_.edges.group, std::vector<hsize_t>{1, 3}, "value", H5T_NATIVE_DOUBLE);
}

void DumpH5MDImpl::open(const std::string& filename,
                        const data::Subdomain& subdomain,
                        const data::Atoms& atoms)
{
    if (state_.fileId >= 0)
    {
        close();
    }

    data::HostAtoms h_atoms(atoms);  // NOLINT

    updateCache(h_atoms);
    state_.saveCount = 0;

    state_.fileId = createFile(filename);

    state_.particleGroupId = createGroup(state_.fileId, "particles");
    state_.particleSubGroupId = createGroup(state_.particleGroupId, config_.particleGroupName);
    writeHeader(state_.fileId);
    openBox(subdomain);

    if (config_.dumpCharge)
    {
        openParticleElement(config_.chargeDataset,
                            std::vector<hsize_t>{1, uint64_c(numLocalParticles), 1},
                            H5T_NATIVE_DOUBLE,
                            state_.charges);
    }

    if (config_.dumpForce)
    {
        openParticleElement(config_.forceDataset,
                            std::vector<hsize_t>{1, uint64_c(numLocalParticles), 3},
                            H5T_NATIVE_DOUBLE,
                            state_.force);
    }

    if (config_.dumpMass)
    {
        openParticleElement(config_.massDataset,
                            std::vector<hsize_t>{1, uint64_c(numLocalParticles), 1},
                            H5T_NATIVE_DOUBLE,
                            state_.mass);
    }

    if (config_.dumpPos)
    {
        openParticleElement(config_.posDataset,
                            std::vector<hsize_t>{1, uint64_c(numLocalParticles), 3},
                            H5T_NATIVE_DOUBLE,
                            state_.pos);
    }

    if (config_.dumpRelativeMass)
    {
        openParticleElement(config_.relativeMassDataset,
                            std::vector<hsize_t>{1, uint64_c(numLocalParticles), 1},
                            H5T_NATIVE_DOUBLE,
                            state_.relativeMass);
    }

    if (config_.dumpType)
    {
        openParticleElement(config_.typeDataset,
                            std::vector<hsize_t>{1, uint64_c(numLocalParticles), 1},
                            H5T_NATIVE_INT64,
                            state_.type);
    }

    if (config_.dumpVel)
    {
        openParticleElement(config_.velDataset,
                            std::vector<hsize_t>{1, uint64_c(numLocalParticles), 3},
                            H5T_NATIVE_DOUBLE,
                            state_.vel);
    }
}

void DumpH5MDImpl::close()
{
    closeParticleElement(state_.vel);
    closeParticleElement(state_.type);
    closeParticleElement(state_.relativeMass);
    closeParticleElement(state_.pos);
    closeParticleElement(state_.mass);
    closeParticleElement(state_.charges);
    closeParticleElement(state_.force);
    closeParticleElement(state_.edges);
    closeGroup(state_.boxGroupId);
    closeGroup(state_.particleSubGroupId);
    closeGroup(state_.particleGroupId);
    closeFile(state_.fileId);
    state_.saveCount = 0;
}

void DumpH5MDImpl::dumpStep(const data::Subdomain& subdomain,
                            const data::Atoms& atoms,
                            const idx_t step,
                            const real_t dt)
{
    data::HostAtoms h_atoms(atoms);  // NOLINT

    updateCache(h_atoms);

    appendEdges(step, dt, subdomain);
    if (config_.dumpCharge)
    {
        appendParticleElement<real_t>(
            step,
            dt,
            h_atoms,
            1,
            [&](const idx_t idx, const int64_t /*dim*/) { return h_atoms.getCharge()(idx); },
            state_.charges);
    }
    if (config_.dumpForce)
    {
        appendParticleElement<real_t>(
            step,
            dt,
            h_atoms,
            3,
            [&](const idx_t idx, const int64_t dim) { return h_atoms.getForce()(idx, dim); },
            state_.force);
    }
    if (config_.dumpMass)
    {
        appendParticleElement<real_t>(
            step,
            dt,
            h_atoms,
            1,
            [&](const idx_t idx, const int64_t /*dim*/) { return h_atoms.getMass()(idx); },
            state_.mass);
    }
    if (config_.dumpPos)
    {
        appendParticleElement<real_t>(
            step,
            dt,
            h_atoms,
            3,
            [&](const idx_t idx, const int64_t dim) { return h_atoms.getPos()(idx, dim); },
            state_.pos);
    }
    if (config_.dumpRelativeMass)
    {
        appendParticleElement<real_t>(
            step,
            dt,
            h_atoms,
            1,
            [&](const idx_t idx, const int64_t /*dim*/) { return h_atoms.getRelativeMass()(idx); },
            state_.relativeMass);
    }
    if (config_.dumpType)
    {
        appendParticleElement<idx_t>(
            step,
            dt,
            h_atoms,
            1,
            [&](const idx_t idx, const int64_t /*dim*/) { return h_atoms.getType()(idx); },
            state_.type);
    }
    if (config_.dumpVel)
    {
        appendParticleElement<real_t>(
            step,
            dt,
            h_atoms,
            3,
            [&](const idx_t idx, const int64_t dim) { return h_atoms.getVel()(idx, dim); },
            state_.vel);
    }
    state_.saveCount += 1;
}

template <typename T>
void DumpH5MDImpl::appendData(const hid_t datasetId,
                              const std::vector<T>& data,
                              const std::vector<hsize_t>& dims) const
{
    const hid_t currentSpace = CHECK_HDF5(H5Dget_space(datasetId));
    const int rank = CHECK_HDF5(H5Sget_simple_extent_ndims(currentSpace));
    MRMD_HOST_CHECK_EQUAL(rank, int_c(dims.size()));

    std::vector<hsize_t> currentSize(dims.size());
    CHECK_HDF5(H5Sget_simple_extent_dims(currentSpace, currentSize.data(), nullptr));
    CHECK_HDF5(H5Sclose(currentSpace));

    std::vector<hsize_t> newSize = currentSize;
    newSize[0] = state_.saveCount + 1;
    for (size_t idx = 1; idx < dims.size(); ++idx)
    {
        newSize[idx] = std::max(newSize[idx], dims[idx]);
    }
    CHECK_HDF5(H5Dset_extent(datasetId, newSize.data()));

    const hid_t fileSpace = CHECK_HDF5(H5Dget_space(datasetId));

    std::vector<hsize_t> offset(dims.size(), 0);
    offset[0] = state_.saveCount;
    std::vector<hsize_t> stride(dims.size(), 1);
    std::vector<hsize_t> count(dims.size(), 1);

    CHECK_HDF5(H5Sselect_hyperslab(
        fileSpace, H5S_SELECT_SET, offset.data(), stride.data(), count.data(), dims.data()));

    std::vector<hsize_t> localOffset(dims.size(), 0);
    const hid_t memorySpace =
        CHECK_HDF5(H5Screate_simple(int_c(dims.size()), dims.data(), nullptr));
    CHECK_HDF5(H5Sselect_hyperslab(
        memorySpace, H5S_SELECT_SET, localOffset.data(), stride.data(), count.data(), dims.data()));

    CHECK_HDF5(
        H5Dwrite(datasetId, typeToHDF5<T>(), memorySpace, fileSpace, H5P_DEFAULT, data.data()));

    CHECK_HDF5(H5Sclose(fileSpace));
    CHECK_HDF5(H5Sclose(memorySpace));
}

void DumpH5MDImpl::appendEdges(const idx_t& step,
                               const real_t& dt,
                               const data::Subdomain& subdomain) const
{
    appendData(state_.edges.step, std::vector<idx_t>{step}, std::vector<hsize_t>{1});
    appendData(state_.edges.time, std::vector<real_t>{real_c(step) * dt}, std::vector<hsize_t>{1});
    appendData(
        state_.edges.value,
        std::vector<real_t>{subdomain.diameter[0], subdomain.diameter[1], subdomain.diameter[2]},
        std::vector<hsize_t>{1, 3});
}

void DumpH5MDImpl::updateCache(const data::HostAtoms& atoms)
{
    numLocalParticles = atoms.numLocalAtoms;
}

void DumpH5MDImpl::dump(const std::string& filename,
                        const data::Subdomain& subdomain,
                        const data::Atoms& atoms)
{
    data::HostAtoms h_atoms(atoms);  // NOLINT

    updateCache(h_atoms);

    auto file_id = CHECK_HDF5(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));

    auto group1 =
        CHECK_HDF5(H5Gcreate(file_id, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    std::string particleGroup = "/particles/" + config_.particleGroupName;
    auto group2 = CHECK_HDF5(
        H5Gcreate(file_id, particleGroup.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    writeHeader(file_id);
    writeBox(file_id, subdomain);
    if (config_.dumpPos)
    {
        writeParticleElement<real_t>(file_id,
                                     config_.posDataset,
                                     h_atoms,
                                     3,
                                     [&](const idx_t idx, const int64_t dim)
                                     { return h_atoms.getPos()(idx, dim); });
    }
    if (config_.dumpVel)
    {
        writeParticleElement<real_t>(file_id,
                                     config_.velDataset,
                                     h_atoms,
                                     3,
                                     [&](const idx_t idx, const int64_t dim)
                                     { return h_atoms.getVel()(idx, dim); });
    }
    if (config_.dumpForce)
    {
        writeParticleElement<real_t>(file_id,
                                     config_.forceDataset,
                                     h_atoms,
                                     3,
                                     [&](const idx_t idx, const int64_t dim)
                                     { return h_atoms.getForce()(idx, dim); });
    }
    if (config_.dumpType)
    {
        writeParticleElement<idx_t>(file_id,
                                    config_.typeDataset,
                                    h_atoms,
                                    1,
                                    [&](const idx_t idx, const int64_t /*dim*/)
                                    { return h_atoms.getType()(idx); });
    }
    if (config_.dumpMass)
    {
        writeParticleElement<real_t>(file_id,
                                     config_.massDataset,
                                     h_atoms,
                                     1,
                                     [&](const idx_t idx, const int64_t /*dim*/)
                                     { return h_atoms.getMass()(idx); });
    }
    if (config_.dumpCharge)
    {
        writeParticleElement<real_t>(file_id,
                                     config_.chargeDataset,
                                     h_atoms,
                                     1,
                                     [&](const idx_t idx, const int64_t /*dim*/)
                                     { return h_atoms.getCharge()(idx); });
    }
    if (config_.dumpRelativeMass)
    {
        writeParticleElement<real_t>(file_id,
                                     config_.relativeMassDataset,
                                     h_atoms,
                                     1,
                                     [&](const idx_t idx, const int64_t /*dim*/)
                                     { return h_atoms.getRelativeMass()(idx); });
    }

    CHECK_HDF5(H5Gclose(group1));
    CHECK_HDF5(H5Gclose(group2));

    CHECK_HDF5(H5Fclose(file_id));
}
}  // namespace impl

void DumpH5MD::open(const std::string& filename,
                    const data::Subdomain& subdomain,
                    const data::Atoms& atoms)
{
    impl::DumpH5MDImpl helper(*this);
    helper.open(filename, subdomain, atoms);
}

void DumpH5MD::dumpStep(const data::Subdomain& subdomain,
                        const data::Atoms& atoms,
                        const idx_t step,
                        const real_t dt)
{
    impl::DumpH5MDImpl helper(*this);
    helper.dumpStep(subdomain, atoms, step, dt);
}

void DumpH5MD::close()
{
    impl::DumpH5MDImpl helper(*this);
    helper.close();
}

void DumpH5MD::dump(const std::string& filename,
                    const data::Subdomain& subdomain,
                    const data::Atoms& atoms)
{
    impl::DumpH5MDImpl helper(*this);
    helper.dump(filename, subdomain, atoms);
}
#else
void DumpH5MD::open(const std::string& /*filename*/,
                    const data::Subdomain& /*subdomain*/,
                    const data::Atoms& /*atoms*/)
{
    MRMD_HOST_CHECK(false, "HDF5 Support not available!");
    exit(EXIT_FAILURE);
}

void DumpH5MD::dumpStep(const data::Subdomain& /*subdomain*/,
                        const data::Atoms& /*atoms*/,
                        const idx_t /*step*/,
                        const real_t /*dt*/)
{
    MRMD_HOST_CHECK(false, "HDF5 Support not available!");
    exit(EXIT_FAILURE);
}

void DumpH5MD::close()
{
    MRMD_HOST_CHECK(false, "HDF5 Support not available!");
    exit(EXIT_FAILURE);
}

void DumpH5MD::dump(const std::string& /*filename*/,
                    const data::Subdomain& /*subdomain*/,
                    const data::Atoms& /*atoms*/)
{
    MRMD_HOST_CHECK(false, "HDF5 Support not available!");
    exit(EXIT_FAILURE);
}
#endif

}  // namespace mrmd::io