#include "DumpH5MDParallel.hpp"

#include <fmt/format.h>

#include <numeric>

#include "assert.hpp"
#include "version.hpp"

namespace mrmd::io
{
template <typename T>
void DumpH5MDParallel::writeParallel(hid_t fileId,
                                     const std::string& name,
                                     const std::vector<hsize_t>& globalDims,
                                     const std::vector<hsize_t>& localDims,
                                     const std::vector<T>& data)
{
    CHECK_EQUAL(globalDims.size(), localDims.size());
    CHECK_EQUAL(
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
        CHECK_LESSEQUAL(localDims[i] + offset[i], globalDims[i], fmt::format("i = {}", i));
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

void DumpH5MDParallel::writeHeader(hid_t fileId) const
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

    CHECK_HDF5(H5LTset_attribute_string(fileId, "/h5md/author", "name", author_.c_str()));

    CHECK_HDF5(H5LTset_attribute_string(fileId, "/h5md/creator", "name", PROJECT_NAME.c_str()));
    CHECK_HDF5(H5LTset_attribute_string(fileId, "/h5md/creator", "version", MRMD_VERSION.c_str()));
}

void DumpH5MDParallel::writeBox(hid_t fileId, const data::Subdomain& subdomain)
{
    std::string groupName = "/particles/" + particleGroupName_ + "/box";
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

    std::vector<double> edges = {
        subdomain.diameter[0], subdomain.diameter[1], subdomain.diameter[2]};
    CHECK_HDF5(
        H5LTset_attribute_double(fileId, groupName.c_str(), "edges", edges.data(), edges.size()));

    CHECK_HDF5(H5Gclose(group));
}

void DumpH5MDParallel::writePos(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 3;  ///< dimensions of the property

    std::string groupName = "/particles/" + particleGroupName_ + "/" + posDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getPos()(idx, 0));
        data.emplace_back(atoms.getPos()(idx, 1));
        data.emplace_back(atoms.getPos()(idx, 2));
    }
    CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

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

void DumpH5MDParallel::writeVel(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 3;  ///< dimensions of the property

    std::string groupName = "/particles/" + particleGroupName_ + "/" + velDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getVel()(idx, 0));
        data.emplace_back(atoms.getVel()(idx, 1));
        data.emplace_back(atoms.getVel()(idx, 2));
    }
    CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

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

void DumpH5MDParallel::writeForce(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 3;  ///< dimensions of the property

    std::string groupName = "/particles/" + particleGroupName_ + "/" + forceDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getForce()(idx, 0));
        data.emplace_back(atoms.getForce()(idx, 1));
        data.emplace_back(atoms.getForce()(idx, 2));
    }
    CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

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

void DumpH5MDParallel::writeType(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = idx_t;
    constexpr int64_t dimensions = 1;  ///< dimensions of the property

    std::string groupName = "/particles/" + particleGroupName_ + "/" + typeDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getType()(idx));
    }
    CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

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

void DumpH5MDParallel::writeMass(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 1;  ///< dimensions of the property

    std::string groupName = "/particles/" + particleGroupName_ + "/" + massDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getMass()(idx));
    }
    CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

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

void DumpH5MDParallel::writeCharge(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 1;  ///< dimensions of the property

    std::string groupName = "/particles/" + particleGroupName_ + "/" + chargeDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getCharge()(idx));
    }
    CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

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

void DumpH5MDParallel::writeRelativeMass(hid_t fileId, const data::HostAtoms& atoms)
{
    using Datatype = real_t;
    constexpr int64_t dimensions = 1;  ///< dimensions of the property

    std::string groupName = "/particles/" + particleGroupName_ + "/" + relativeMassDataset;
    auto group = H5Gcreate(fileId, groupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<Datatype> data;
    data.reserve(numLocalParticles * dimensions);
    for (idx_t idx = 0; idx < atoms.numLocalAtoms; ++idx)
    {
        data.emplace_back(atoms.getRelativeMass()(idx));
    }
    CHECK_EQUAL(int64_c(data.size()), numLocalParticles * dimensions);

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

void DumpH5MDParallel::updateCache(const data::HostAtoms& atoms)
{
    numLocalParticles = atoms.numLocalAtoms;
    MPI_Allreduce(reinterpret_cast<const void*>(&numLocalParticles),
                  reinterpret_cast<void*>(&numTotalParticles),
                  1,
                  MPI_INT64_T,
                  MPI_SUM,
                  mpiInfo_->comm);

    MPI_Exscan(&numLocalParticles, &particleOffset, 1, MPI_INT64_T, MPI_SUM, mpiInfo_->comm);
    if (mpiInfo_->rank == 0) particleOffset = 0;
}

void DumpH5MDParallel::dump(const std::string& filename,
                            const data::Subdomain& subdomain,
                            const data::Atoms& atoms)
{
    data::HostAtoms h_atoms(atoms);

    updateCache(h_atoms);

    auto info = MPI_INFO_NULL;

    auto plist = CHECK_HDF5(H5Pcreate(H5P_FILE_ACCESS));
    CHECK_HDF5(H5Pset_fapl_mpio(plist, mpiInfo_->comm, info));

    auto file_id = CHECK_HDF5(H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist));

    auto group1 =
        CHECK_HDF5(H5Gcreate(file_id, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    std::string particleGroup = "/particles/" + particleGroupName_;
    auto group2 = CHECK_HDF5(
        H5Gcreate(file_id, particleGroup.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

    writeHeader(file_id);
    writeBox(file_id, subdomain);
    if (dumpPos) writePos(file_id, h_atoms);
    if (dumpVel) writeVel(file_id, h_atoms);
    if (dumpForce) writeForce(file_id, h_atoms);
    if (dumpType) writeType(file_id, h_atoms);
    if (dumpMass) writeMass(file_id, h_atoms);
    if (dumpCharge) writeCharge(file_id, h_atoms);
    if (dumpRelativeMass) writeRelativeMass(file_id, h_atoms);

    CHECK_HDF5(H5Gclose(group1));
    CHECK_HDF5(H5Gclose(group2));

    CHECK_HDF5(H5Fclose(file_id));
}

}  // namespace mrmd::io