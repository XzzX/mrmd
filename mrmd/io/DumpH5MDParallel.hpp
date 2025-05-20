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

#include "data/Atoms.hpp"
#include "data/MPIInfo.hpp"
#include "data/Subdomain.hpp"

namespace mrmd::io
{

class DumpH5MDParallel
{
public:
    DumpH5MDParallel(const std::shared_ptr<data::MPIInfo>& mpiInfoArg,
                     const std::string& authorArg,
                     const std::string& particleSubGroupNameArg = "atoms")
        : mpiInfo(mpiInfoArg), author(authorArg), particleSubGroupName(particleSubGroupNameArg)
    {
    }
    void open(const std::string& filename);

    void dumpStep(
        const data::Subdomain& subdomain,
        const data::Atoms& atoms,
        const idx_t step,
        const real_t dt);

    void close();

    void dump(const std::string& filename,
              const data::Subdomain& subdomain,
              const data::Atoms& atoms);

    bool dumpPos = true;
    bool dumpVel = true;
    bool dumpForce = true;
    bool dumpType = true;
    bool dumpMass = true;
    bool dumpCharge = true;
    bool dumpRelativeMass = true;

    std::string posDataset = "position";
    std::string velDataset = "velocity";
    std::string forceDataset = "force";
    std::string typeDataset = "type";
    std::string massDataset = "mass";
    std::string chargeDataset = "charge";
    std::string relativeMassDataset = "relativeMass";

    std::shared_ptr<data::MPIInfo> mpiInfo;

    std::string author = "xxx";
    std::string particleSubGroupName = "atoms";

    hid_t fileId;
    hid_t particleGroupId;
    hid_t particleSubGroupId;
    hid_t boxGroupId;
    hid_t edgesGroupId;
    hid_t stepSetId;
    hid_t timeSetId;
    hid_t boxValueSetId;

    hsize_t saveCount = 0;
};
}  // namespace mrmd::io