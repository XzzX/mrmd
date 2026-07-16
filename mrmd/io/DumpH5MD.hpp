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
namespace impl
{
class DumpH5MDImpl;
}

class DumpH5MD
{
public:
    DumpH5MD(const std::string& authorArg, const std::string& particleGroupNameArg = "atoms");

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

private:
    friend class impl::DumpH5MDImpl;

    struct ElementHandles
    {
        int64_t group = -1;
        int64_t step = -1;
        int64_t time = -1;
        int64_t value = -1;
    };

    struct State
    {
        int64_t fileId = -1;
        int64_t particleGroupId = -1;
        int64_t particleSubGroupId = -1;
        int64_t boxGroupId = -1;
        ElementHandles edges;
        ElementHandles charges;
        ElementHandles force;
        ElementHandles mass;
        ElementHandles pos;
        ElementHandles relativeMass;
        ElementHandles type;
        ElementHandles vel;
        uint64_t saveCount = 0;
    };

    State state_;
};

}  // namespace mrmd::io