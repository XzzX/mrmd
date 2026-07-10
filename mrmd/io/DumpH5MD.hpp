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

#pragma once

#include <string>

#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"

namespace mrmd::io
{
class DumpH5MD
{
public:
    DumpH5MD(const std::string& authorArg, const std::string& particleGroupNameArg = "atoms")
        : author(authorArg), particleGroupName(particleGroupNameArg)
    {
    }
    DumpH5MD(const DumpH5MD&) = delete;
    DumpH5MD& operator=(const DumpH5MD&) = delete;
    DumpH5MD(DumpH5MD&&) = delete;
    DumpH5MD& operator=(DumpH5MD&&) = delete;

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

    std::string author = "xxx";
    std::string particleGroupName = "atoms";

    int64_t fileId = -1;
    int64_t particleGroupId = -1;
    int64_t particleSubGroupId = -1;
    int64_t boxGroupId = -1;
    int64_t edgesGroupId = -1;
    int64_t edgesStepSetId = -1;
    int64_t edgesTimeSetId = -1;
    int64_t edgesValueSetId = -1;
    int64_t chargesGroupId = -1;
    int64_t chargesStepSetId = -1;
    int64_t chargesTimeSetId = -1;
    int64_t chargesValueSetId = -1;
    int64_t forceGroupId = -1;
    int64_t forceStepSetId = -1;
    int64_t forceTimeSetId = -1;
    int64_t forceValueSetId = -1;
    int64_t massGroupId = -1;
    int64_t massStepSetId = -1;
    int64_t massTimeSetId = -1;
    int64_t massValueSetId = -1;
    int64_t posGroupId = -1;
    int64_t posStepSetId = -1;
    int64_t posTimeSetId = -1;
    int64_t posValueSetId = -1;
    int64_t relativeMassGroupId = -1;
    int64_t relativeMassStepSetId = -1;
    int64_t relativeMassTimeSetId = -1;
    int64_t relativeMassValueSetId = -1;
    int64_t typeGroupId = -1;
    int64_t typeStepSetId = -1;
    int64_t typeTimeSetId = -1;
    int64_t typeValueSetId = -1;
    int64_t velGroupId = -1;
    int64_t velStepSetId = -1;
    int64_t velTimeSetId = -1;
    int64_t velValueSetId = -1;

    uint64_t saveCount = 0;
};

}  // namespace mrmd::io