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
                     const std::string& particleGroupNameArg = "atoms")
        : mpiInfo(mpiInfoArg),
          author(authorArg),
          particleGroupName(particleGroupNameArg)
    {
    }

    void dump(const std::string& filename, const data::Subdomain& subdomain, const data::Atoms& atoms);

    
    bool dumpPos = true;
    bool dumpVel = true;
    bool dumpForce = true;
    bool dumpType = true;
    bool dumpMass = true;
    bool dumpCharge = true;
    bool dumpRelativeMass = true;

    
    std::string posDataset = "pos";
    std::string velDataset = "vel";
    std::string forceDataset = "force";
    std::string typeDataset = "type";
    std::string massDataset = "mass";
    std::string chargeDataset = "charge";
    std::string relativeMassDataset = "relativeMass";

    std::shared_ptr<data::MPIInfo> mpiInfo;

    std::string author = "xxx";
    std::string particleGroupName = "atoms";
};

}  // namespace mrmd::io