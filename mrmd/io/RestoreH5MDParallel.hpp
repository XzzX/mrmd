#pragma once

#include <mpi.h>

#include <memory>
#include <string>

#include "data/Atoms.hpp"
#include "data/MPIInfo.hpp"
#include "data/Subdomain.hpp"
#include "hdf5.hpp"

namespace mrmd::io
{

class RestoreH5MDParallel
{
public:
    RestoreH5MDParallel(const std::shared_ptr<data::MPIInfo>& mpiInfo,
                        const std::string& particleGroupName = "atoms")
        : mpiInfo_(mpiInfo), particleGroupName_(particleGroupName)
    {
    }

    void restore(const std::string& filename, data::Subdomain& subdomain, data::Atoms& atoms);

    bool restorePos = true;
    bool restoreVel = true;
    bool restoreForce = true;
    bool restoreType = true;
    bool restoreMass = true;
    bool restoreCharge = true;
    bool restoreRelativeMass = true;

    std::string posDataset = "pos";
    std::string velDataset = "vel";
    std::string forceDataset = "force";
    std::string typeDataset = "type";
    std::string massDataset = "mass";
    std::string chargeDataset = "charge";
    std::string relativeMassDataset = "relativeMass";

private:
    template <typename T>
    void readParallel(hid_t fileId, const std::string& dataset, std::vector<T>& data);

    std::shared_ptr<data::MPIInfo> mpiInfo_;
    std::string particleGroupName_;
};

}  // namespace mrmd::io