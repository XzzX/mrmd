#pragma once

#include <mpi.h>

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
    DumpH5MDParallel(const std::shared_ptr<data::MPIInfo>& mpiInfo,
                     const std::string& author,
                     const std::string& particleGroupName = "atoms")
        : mpiInfo_(mpiInfo), author_(author), particleGroupName_(particleGroupName)
    {
    }

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

    std::string posDataset = "pos";
    std::string velDataset = "vel";
    std::string forceDataset = "force";
    std::string typeDataset = "type";
    std::string massDataset = "mass";
    std::string chargeDataset = "charge";
    std::string relativeMassDataset = "relativeMass";

private:
    std::shared_ptr<data::MPIInfo> mpiInfo_;

    std::string author_ = "xxx";
    std::string particleGroupName_ = "atoms";
};

}  // namespace mrmd::io