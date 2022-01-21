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
    void updateCache(const data::HostAtoms& atoms);

    void writeHeader(hid_t fileId) const;
    void writeBox(hid_t fileId, const data::Subdomain& subdomain);
    void writePos(hid_t fileId, const data::HostAtoms& atoms);
    void writeVel(hid_t fileId, const data::HostAtoms& atoms);
    void writeForce(hid_t fileId, const data::HostAtoms& atoms);
    void writeType(hid_t fileId, const data::HostAtoms& atoms);
    void writeMass(hid_t fileId, const data::HostAtoms& atoms);
    void writeCharge(hid_t fileId, const data::HostAtoms& atoms);
    void writeRelativeMass(hid_t fileId, const data::HostAtoms& atoms);

    template <typename T>
    void writeParallel(hid_t fileId,
                       const std::string& name,
                       const std::vector<hsize_t>& globalDims,
                       const std::vector<hsize_t>& localDims,
                       const std::vector<T>& data);

    std::shared_ptr<data::MPIInfo> mpiInfo_;

    std::string author_ = "xxx";
    std::string particleGroupName_ = "atoms";

    int64_t numLocalParticles = -1;
    int64_t numTotalParticles = -1;
    /// Offset of the local particle chunk in the global particle array.
    int64_t particleOffset = -1;
};

}  // namespace mrmd::io