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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include "DumpH5MD.hpp"
#include "RestoreH5MD.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "hdf5.hpp"

namespace mrmd
{
namespace io
{
#ifdef MRMD_ENABLE_HDF5
namespace
{
std::vector<hsize_t> getDatasetExtents(const std::string& filename, const std::string& datasetPath)
{
    const hid_t fileId = CHECK_HDF5(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    const hid_t datasetId = CHECK_HDF5(H5Dopen(fileId, datasetPath.c_str(), H5P_DEFAULT));
    const hid_t dataspaceId = CHECK_HDF5(H5Dget_space(datasetId));

    const int rank = CHECK_HDF5(H5Sget_simple_extent_ndims(dataspaceId));
    std::vector<hsize_t> dims(uint64_c(rank), 0);
    CHECK_HDF5(H5Sget_simple_extent_dims(dataspaceId, dims.data(), nullptr));

    CHECK_HDF5(H5Sclose(dataspaceId));
    CHECK_HDF5(H5Dclose(datasetId));
    CHECK_HDF5(H5Fclose(fileId));

    return dims;
}

template <typename T>
std::vector<T> readDataset1D(const std::string& filename,
                             const std::string& datasetPath,
                             const hsize_t expectedSize)
{
    std::vector<T> data(expectedSize);
    const hid_t fileId = CHECK_HDF5(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    CHECK_HDF5(H5LTread_dataset(fileId, datasetPath.c_str(), typeToHDF5<T>(), data.data()));
    CHECK_HDF5(H5Fclose(fileId));
    return data;
}
}  // namespace
#endif

data::Atoms getAtoms()
{
    auto atoms = data::Atoms(10);
    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto force = atoms.getForce();
    auto type = atoms.getType();
    auto charge = atoms.getCharge();
    auto mass = atoms.getMass();
    auto relativeMass = atoms.getRelativeMass();

    auto policy = Kokkos::RangePolicy<>(0, 10);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx)
    {
        pos(idx, 0) = real_c(idx);
        pos(idx, 1) = real_c(idx) + 0.1_r;
        pos(idx, 2) = real_c(idx) + 0.2_r;

        vel(idx, 0) = real_c(idx + 1);
        vel(idx, 1) = real_c(idx + 1) + 0.1_r;
        vel(idx, 2) = real_c(idx + 1) + 0.2_r;

        force(idx, 0) = real_c(idx + 2);
        force(idx, 1) = real_c(idx + 2) + 0.1_r;
        force(idx, 2) = real_c(idx + 2) + 0.2_r;

        type(idx) = idx + 3;
        charge(idx) = real_c(idx) + 4.1_r;
        mass(idx) = real_c(idx) + 5.2_r;
        relativeMass(idx) = real_c(idx) + 6.3_r;
    };
    Kokkos::parallel_for("init-atoms", policy, kernel);
    Kokkos::fence();

    atoms.numLocalAtoms = 10;
    atoms.numGhostAtoms = 0;

    return atoms;
}
TEST(H5MD, dump)
{
    auto subdomain1 = data::Subdomain({1_r, 2_r, 3_r}, {4_r, 6_r, 8_r}, 0.5_r);
    auto atoms1 = getAtoms();

    auto dump = DumpH5MD("XzzX");
    dump.dump("dummy.h5md", subdomain1, atoms1);

    auto subdomain2 = data::Subdomain();
    auto atoms2 = data::Atoms(0);
    auto restore = RestoreH5MD();
    restore.restore("dummy.h5md", subdomain2, atoms2);

    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[0], subdomain2.ghostLayerThickness[0]);
    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[1], subdomain2.ghostLayerThickness[1]);
    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[2], subdomain2.ghostLayerThickness[2]);

    EXPECT_FLOAT_EQ(subdomain1.minCorner[0], subdomain2.minCorner[0]);
    EXPECT_FLOAT_EQ(subdomain1.minCorner[1], subdomain2.minCorner[1]);
    EXPECT_FLOAT_EQ(subdomain1.minCorner[2], subdomain2.minCorner[2]);

    EXPECT_FLOAT_EQ(subdomain1.maxCorner[0], subdomain2.maxCorner[0]);
    EXPECT_FLOAT_EQ(subdomain1.maxCorner[1], subdomain2.maxCorner[1]);
    EXPECT_FLOAT_EQ(subdomain1.maxCorner[2], subdomain2.maxCorner[2]);

    auto h_atoms1 = data::HostAtoms(atoms1);  // NOLINT
    auto h_atoms2 = data::HostAtoms(atoms2);  // NOLINT
    EXPECT_EQ(h_atoms1.numLocalAtoms, h_atoms2.numLocalAtoms);
    EXPECT_EQ(h_atoms1.numGhostAtoms, h_atoms2.numGhostAtoms);
    for (idx_t idx = 0; idx < h_atoms2.numLocalAtoms; ++idx)
    {
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 0), h_atoms2.getPos()(idx, 0));
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 1), h_atoms2.getPos()(idx, 1));
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 2), h_atoms2.getPos()(idx, 2));

        EXPECT_FLOAT_EQ(h_atoms1.getVel()(idx, 0), h_atoms2.getVel()(idx, 0));
        EXPECT_FLOAT_EQ(h_atoms1.getVel()(idx, 1), h_atoms2.getVel()(idx, 1));
        EXPECT_FLOAT_EQ(h_atoms1.getVel()(idx, 2), h_atoms2.getVel()(idx, 2));

        EXPECT_FLOAT_EQ(h_atoms1.getForce()(idx, 0), h_atoms2.getForce()(idx, 0));
        EXPECT_FLOAT_EQ(h_atoms1.getForce()(idx, 1), h_atoms2.getForce()(idx, 1));
        EXPECT_FLOAT_EQ(h_atoms1.getForce()(idx, 2), h_atoms2.getForce()(idx, 2));

        EXPECT_EQ(h_atoms1.getType()(idx), h_atoms2.getType()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getCharge()(idx), h_atoms2.getCharge()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getMass()(idx), h_atoms2.getMass()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getRelativeMass()(idx), h_atoms2.getRelativeMass()(idx));
    }
}

TEST(H5MD, dumpStepWithCustomDatasetsAndFlags)
{
    const std::string filename = "dummy_step.h5md";
    auto subdomain1 = data::Subdomain({1_r, 2_r, 3_r}, {4_r, 6_r, 8_r}, 0.5_r);
    auto atoms1 = getAtoms();

    auto dump = DumpH5MD("XzzX");
    dump.dumpVel = false;
    dump.dumpForce = false;
    dump.posDataset = "custom_position";
    dump.typeDataset = "custom_type";
    dump.massDataset = "custom_mass";
    dump.chargeDataset = "custom_charge";
    dump.relativeMassDataset = "custom_relative_mass";

    dump.open(filename, subdomain1, atoms1);
    dump.dumpStep(subdomain1, atoms1, 0, 0.001_r);
    dump.dumpStep(subdomain1, atoms1, 7, 0.001_r);
    dump.close();

#ifdef MRMD_ENABLE_HDF5
    const auto positionValueExtents =
        getDatasetExtents(filename, "/particles/atoms/" + dump.posDataset + "/value");
    ASSERT_EQ(positionValueExtents.size(), 3);
    EXPECT_EQ(positionValueExtents[0], 2);
    EXPECT_EQ(positionValueExtents[1], 10);
    EXPECT_EQ(positionValueExtents[2], 3);

    const auto positionSteps =
        readDataset1D<int64_t>(filename, "/particles/atoms/" + dump.posDataset + "/step", 2);
    ASSERT_EQ(positionSteps.size(), 2);
    EXPECT_EQ(positionSteps[0], 0);
    EXPECT_EQ(positionSteps[1], 7);

    const auto positionTimes =
        readDataset1D<real_t>(filename, "/particles/atoms/" + dump.posDataset + "/time", 2);
    ASSERT_EQ(positionTimes.size(), 2);
    EXPECT_FLOAT_EQ(positionTimes[0], 0._r);
    EXPECT_FLOAT_EQ(positionTimes[1], 0.007_r);
#endif

    auto subdomain2 = data::Subdomain();
    auto atoms2 = data::Atoms(0);
    auto restore = RestoreH5MD();
    restore.restoreVel = false;
    restore.restoreForce = false;
    restore.posDataset = dump.posDataset;
    restore.typeDataset = dump.typeDataset;
    restore.massDataset = dump.massDataset;
    restore.chargeDataset = dump.chargeDataset;
    restore.relativeMassDataset = dump.relativeMassDataset;
    restore.restore(filename, subdomain2, atoms2);

    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[0], subdomain2.ghostLayerThickness[0]);
    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[1], subdomain2.ghostLayerThickness[1]);
    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[2], subdomain2.ghostLayerThickness[2]);

    EXPECT_FLOAT_EQ(subdomain1.minCorner[0], subdomain2.minCorner[0]);
    EXPECT_FLOAT_EQ(subdomain1.minCorner[1], subdomain2.minCorner[1]);
    EXPECT_FLOAT_EQ(subdomain1.minCorner[2], subdomain2.minCorner[2]);

    EXPECT_FLOAT_EQ(subdomain1.maxCorner[0], subdomain2.maxCorner[0]);
    EXPECT_FLOAT_EQ(subdomain1.maxCorner[1], subdomain2.maxCorner[1]);
    EXPECT_FLOAT_EQ(subdomain1.maxCorner[2], subdomain2.maxCorner[2]);

    auto h_atoms1 = data::HostAtoms(atoms1);  // NOLINT
    auto h_atoms2 = data::HostAtoms(atoms2);  // NOLINT
    EXPECT_EQ(h_atoms1.numLocalAtoms, h_atoms2.numLocalAtoms);
    EXPECT_EQ(h_atoms1.numGhostAtoms, h_atoms2.numGhostAtoms);
    for (idx_t idx = 0; idx < h_atoms2.numLocalAtoms; ++idx)
    {
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 0), h_atoms2.getPos()(idx, 0));
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 1), h_atoms2.getPos()(idx, 1));
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 2), h_atoms2.getPos()(idx, 2));

        EXPECT_EQ(h_atoms1.getType()(idx), h_atoms2.getType()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getCharge()(idx), h_atoms2.getCharge()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getMass()(idx), h_atoms2.getMass()(idx));
        EXPECT_FLOAT_EQ(h_atoms1.getRelativeMass()(idx), h_atoms2.getRelativeMass()(idx));
    }
}

TEST(H5MD, dumpStepRoundtripWithCustomDatasetNames)
{
    const std::string filename = "dummy_stream.h5md";
    auto subdomain1 = data::Subdomain({1_r, 2_r, 3_r}, {4_r, 6_r, 8_r}, 0.5_r);
    auto atoms1 = getAtoms();

    auto dump = DumpH5MD("XzzX");
    dump.dumpVel = false;
    dump.dumpForce = false;
    dump.dumpType = false;
    dump.dumpMass = false;
    dump.dumpRelativeMass = false;
    dump.posDataset = "pos_custom";
    dump.chargeDataset = "charge_custom";

    dump.open(filename, subdomain1, atoms1);
    dump.dumpStep(subdomain1, atoms1, 17, 0.25_r);
    dump.close();

#ifdef MRMD_ENABLE_HDF5
    const auto edgesValueExtents = getDatasetExtents(filename, "/particles/atoms/box/edges/value");
    ASSERT_EQ(edgesValueExtents.size(), 2);
    EXPECT_EQ(edgesValueExtents[0], 1);
    EXPECT_EQ(edgesValueExtents[1], 3);

    const auto edgesSteps = readDataset1D<int64_t>(filename, "/particles/atoms/box/edges/step", 1);
    ASSERT_EQ(edgesSteps.size(), 1);
    EXPECT_EQ(edgesSteps[0], 17);

    const auto edgesTimes = readDataset1D<real_t>(filename, "/particles/atoms/box/edges/time", 1);
    ASSERT_EQ(edgesTimes.size(), 1);
    EXPECT_FLOAT_EQ(edgesTimes[0], 4.25_r);

    const auto chargeValueExtents =
        getDatasetExtents(filename, "/particles/atoms/" + dump.chargeDataset + "/value");
    ASSERT_EQ(chargeValueExtents.size(), 3);
    EXPECT_EQ(chargeValueExtents[0], 1);
    EXPECT_EQ(chargeValueExtents[1], 10);
    EXPECT_EQ(chargeValueExtents[2], 1);

    const auto chargeSteps =
        readDataset1D<int64_t>(filename, "/particles/atoms/" + dump.chargeDataset + "/step", 1);
    ASSERT_EQ(chargeSteps.size(), 1);
    EXPECT_EQ(chargeSteps[0], 17);
#endif

    auto subdomain2 = data::Subdomain();
    auto atoms2 = data::Atoms(0);
    auto restore = RestoreH5MD();
    restore.restoreVel = false;
    restore.restoreForce = false;
    restore.restoreType = false;
    restore.restoreMass = false;
    restore.restoreRelativeMass = false;
    restore.posDataset = "pos_custom";
    restore.chargeDataset = "charge_custom";
    restore.restore(filename, subdomain2, atoms2);

    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[0], subdomain2.ghostLayerThickness[0]);
    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[1], subdomain2.ghostLayerThickness[1]);
    EXPECT_FLOAT_EQ(subdomain1.ghostLayerThickness[2], subdomain2.ghostLayerThickness[2]);

    EXPECT_FLOAT_EQ(subdomain1.minCorner[0], subdomain2.minCorner[0]);
    EXPECT_FLOAT_EQ(subdomain1.minCorner[1], subdomain2.minCorner[1]);
    EXPECT_FLOAT_EQ(subdomain1.minCorner[2], subdomain2.minCorner[2]);

    EXPECT_FLOAT_EQ(subdomain1.maxCorner[0], subdomain2.maxCorner[0]);
    EXPECT_FLOAT_EQ(subdomain1.maxCorner[1], subdomain2.maxCorner[1]);
    EXPECT_FLOAT_EQ(subdomain1.maxCorner[2], subdomain2.maxCorner[2]);

    auto h_atoms1 = data::HostAtoms(atoms1);  // NOLINT
    auto h_atoms2 = data::HostAtoms(atoms2);  // NOLINT
    EXPECT_EQ(h_atoms1.numLocalAtoms, h_atoms2.numLocalAtoms);
    EXPECT_EQ(h_atoms1.numGhostAtoms, h_atoms2.numGhostAtoms);
    for (idx_t idx = 0; idx < h_atoms2.numLocalAtoms; ++idx)
    {
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 0), h_atoms2.getPos()(idx, 0));
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 1), h_atoms2.getPos()(idx, 1));
        EXPECT_FLOAT_EQ(h_atoms1.getPos()(idx, 2), h_atoms2.getPos()(idx, 2));
        EXPECT_FLOAT_EQ(h_atoms1.getCharge()(idx), h_atoms2.getCharge()(idx));
    }
}

TEST(H5MD, restoreReadsSingleSelectedFrame)
{
    const std::string filename = "dummy_single_frame_restore.h5md";
    auto subdomain = data::Subdomain({1_r, 2_r, 3_r}, {4_r, 6_r, 8_r}, 0.5_r);
    auto atomsFirstFrame = getAtoms();
    auto atomsSecondFrame = getAtoms();

    {
        auto h_atoms = data::HostAtoms(atomsSecondFrame);  // NOLINT
        for (idx_t idx = 0; idx < h_atoms.numLocalAtoms; ++idx)
        {
            h_atoms.getPos()(idx, 0) += 100._r;
            h_atoms.getPos()(idx, 1) += 100._r;
            h_atoms.getPos()(idx, 2) += 100._r;
            h_atoms.getCharge()(idx) += 10._r;
        }
        data::deep_copy(atomsSecondFrame, h_atoms);
    }

    auto dump = DumpH5MD("XzzX");
    dump.dumpVel = false;
    dump.dumpForce = false;
    dump.dumpType = false;
    dump.dumpMass = false;
    dump.dumpRelativeMass = false;
    dump.posDataset = "pos_custom";
    dump.chargeDataset = "charge_custom";

    dump.open(filename, subdomain, atomsFirstFrame);
    dump.dumpStep(subdomain, atomsFirstFrame, 10, 0.5_r);
    dump.dumpStep(subdomain, atomsSecondFrame, 11, 0.5_r);
    dump.close();

    auto restoredSubdomainLast = data::Subdomain();
    auto restoredAtomsLast = data::Atoms(0);
    auto restoreLast = RestoreH5MD();
    restoreLast.restoreVel = false;
    restoreLast.restoreForce = false;
    restoreLast.restoreType = false;
    restoreLast.restoreMass = false;
    restoreLast.restoreRelativeMass = false;
    restoreLast.posDataset = dump.posDataset;
    restoreLast.chargeDataset = dump.chargeDataset;
    restoreLast.restore(filename, restoredSubdomainLast, restoredAtomsLast);

    auto h_expectedLast = data::HostAtoms(atomsSecondFrame);   // NOLINT
    auto h_restoredLast = data::HostAtoms(restoredAtomsLast);  // NOLINT
    ASSERT_EQ(h_expectedLast.numLocalAtoms, h_restoredLast.numLocalAtoms);
    for (idx_t idx = 0; idx < h_restoredLast.numLocalAtoms; ++idx)
    {
        EXPECT_FLOAT_EQ(h_expectedLast.getPos()(idx, 0), h_restoredLast.getPos()(idx, 0));
        EXPECT_FLOAT_EQ(h_expectedLast.getPos()(idx, 1), h_restoredLast.getPos()(idx, 1));
        EXPECT_FLOAT_EQ(h_expectedLast.getPos()(idx, 2), h_restoredLast.getPos()(idx, 2));
        EXPECT_FLOAT_EQ(h_expectedLast.getCharge()(idx), h_restoredLast.getCharge()(idx));
    }

    auto restoredSubdomainFirst = data::Subdomain();
    auto restoredAtomsFirst = data::Atoms(0);
    auto restoreFirst = RestoreH5MD();
    restoreFirst.restoreVel = false;
    restoreFirst.restoreForce = false;
    restoreFirst.restoreType = false;
    restoreFirst.restoreMass = false;
    restoreFirst.restoreRelativeMass = false;
    restoreFirst.posDataset = dump.posDataset;
    restoreFirst.chargeDataset = dump.chargeDataset;
    restoreFirst.restore(filename, restoredSubdomainFirst, restoredAtomsFirst, 0);

    auto h_expectedFirst = data::HostAtoms(atomsFirstFrame);   // NOLINT
    auto h_restoredFirst = data::HostAtoms(restoredAtomsFirst);  // NOLINT
    ASSERT_EQ(h_expectedFirst.numLocalAtoms, h_restoredFirst.numLocalAtoms);
    for (idx_t idx = 0; idx < h_restoredFirst.numLocalAtoms; ++idx)
    {
        EXPECT_FLOAT_EQ(h_expectedFirst.getPos()(idx, 0), h_restoredFirst.getPos()(idx, 0));
        EXPECT_FLOAT_EQ(h_expectedFirst.getPos()(idx, 1), h_restoredFirst.getPos()(idx, 1));
        EXPECT_FLOAT_EQ(h_expectedFirst.getPos()(idx, 2), h_restoredFirst.getPos()(idx, 2));
        EXPECT_FLOAT_EQ(h_expectedFirst.getCharge()(idx), h_restoredFirst.getCharge()(idx));
    }
}
}  // namespace io
}  // namespace mrmd