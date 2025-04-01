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
    void open(const std::string& filename, const data::Subdomain& subdomain, const data::Atoms& atoms);

    void dumpStep(const data::Subdomain& subdomain,
                  const data::Atoms& atoms,
                  const idx_t& step,
                  const real_t& dt);

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

    int64_t fileId;
    int64_t particleGroupId;
    int64_t particleSubGroupId;
    int64_t boxGroupId;
    int64_t edgesGroupId;
    int64_t edgesStepSetId;
    int64_t edgesTimeSetId;
    int64_t edgesValueSetId;
    int64_t chargesGroupId;
    int64_t chargesStepSetId;
    int64_t chargesTimeSetId;
    int64_t chargesValueSetId;
    int64_t forceGroupId;
    int64_t forceStepSetId;
    int64_t forceTimeSetId;
    int64_t forceValueSetId;
    int64_t massGroupId;
    int64_t massStepSetId;
    int64_t massTimeSetId;
    int64_t massValueSetId;
    int64_t posGroupId;
    int64_t posStepSetId;
    int64_t posTimeSetId;
    int64_t posValueSetId;
    int64_t relativeMassGroupId;
    int64_t relativeMassStepSetId;
    int64_t relativeMassTimeSetId;
    int64_t relativeMassValueSetId;
    int64_t typeGroupId;
    int64_t typeStepSetId;
    int64_t typeTimeSetId;
    int64_t typeValueSetId;
    int64_t velGroupId;
    int64_t velStepSetId;
    int64_t velTimeSetId;
    int64_t velValueSetId;

    uint64_t saveCount = 0;
};
}  // namespace mrmd::io