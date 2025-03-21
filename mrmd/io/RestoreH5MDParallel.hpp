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

#include <memory>
#include <string>

#include "cmake.hpp"
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
                        const std::string& particleSubGroupName = "atoms")
        : mpiInfo_(mpiInfo), particleSubGroupName_(particleSubGroupName)
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

    std::string posDataset = "position";
    std::string velDataset = "velocity";
    std::string forceDataset = "force";
    std::string typeDataset = "type";
    std::string massDataset = "mass";
    std::string chargeDataset = "charge";
    std::string relativeMassDataset = "relativeMass";

private:
    template <typename T>
    void readParallel(hid_t fileId, const std::string& dataset, std::vector<T>& data);

    std::shared_ptr<data::MPIInfo> mpiInfo_;
    std::string particleSubGroupName_;
};

}  // namespace mrmd::io