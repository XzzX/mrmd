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

#include "NVT.hpp"

#include <format>

#include <algorithm>

#include "action/BerendsenThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "communication/GhostLayer.hpp"
#include "io/DumpCSV.hpp"
#include "util/ExponentialMovingAverage.hpp"
#include "util/PrintTable.hpp"

namespace mrmd
{
void nvt(YAML::Node& config, data::Atoms& atoms, const data::Subdomain& subdomain)
{
    constexpr int64_t estimatedMaxNeighbors = 60;
    constexpr real_t cellRatio = 0.5_r;
    const real_t skin = config["LJ"]["skin"].as<real_t>();
    auto rcVec = config["LJ"]["cutoff"].as<std::vector<real_t>>();
    const real_t rc = std::ranges::max(rcVec);
    const real_t neighborCutoff = rc + skin;
    auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];

    communication::GhostLayer ghostLayer;
    auto LJ = action::LennardJones(config["LJ"]["capping"].as<std::vector<real_t>>(),
                                   config["LJ"]["cutoff"].as<std::vector<real_t>>(),
                                   config["LJ"]["sigma"].as<std::vector<real_t>>(),
                                   config["LJ"]["epsilon"].as<std::vector<real_t>>(),
                                   2,
                                   true);
    HalfVerletList verletList;

    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    util::ExponentialMovingAverage currentPressure(
        config["pressure_averaging_coefficient"].as<real_t>());
    util::ExponentialMovingAverage currentTemperature(
        config["temperature_averaging_coefficient"].as<real_t>());
    currentTemperature << analysis::getMeanKineticEnergy(atoms) * 2_r / 3_r;

    Kokkos::Timer timer;

    if (config["enable_output"].as<bool>())
        util::printTable(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
    if (config["enable_output"].as<bool>())
        util::printTableSep(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
    for (auto step = 0; step < config["time_steps"].as<int64_t>(); ++step)
    {
        maxAtomDisplacement +=
            action::VelocityVerlet::preForceIntegrate(atoms, config["dt"].as<real_t>());

        if (step % config["thermostat_interval"].as<int64_t>() == 0)
        {
            action::BerendsenThermostat::apply(
                atoms,
                currentTemperature,
                config["target_temperature"].as<real_t>(),
                config["temperature_relaxation_coefficient"].as<real_t>());
        }

        if (maxAtomDisplacement >= skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            //            real_t gridDelta[3] = {neighborCutoff, neighborCutoff, neighborCutoff};
            //            LinkedCellList linkedCellList(atoms.getPos(),
            //                                          0,
            //                                          atoms.numLocalAtoms,
            //                                          gridDelta,
            //                                          subdomain.minCorner.data(),
            //                                          subdomain.maxCorner.data());
            //            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(atoms, subdomain);
            verletList.build(atoms.getPos(),
                             0,
                             atoms.numLocalAtoms,
                             neighborCutoff,
                             cellRatio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data(),
                             estimatedMaxNeighbors);
        }
        else
        {
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        auto force = atoms.getForce();
        Cabana::deep_copy(force, 0_r);

        LJ.apply(atoms, verletList);

        if (step < 201)
        {
            currentPressure = util::ExponentialMovingAverage(
                config["pressure_averaging_coefficient"].as<real_t>());
            currentTemperature = util::ExponentialMovingAverage(
                config["temperature_averaging_coefficient"].as<real_t>());
        }
        auto Ek = analysis::getKineticEnergy(atoms);
        currentPressure << 2_r * (Ek - LJ.getVirial()) / (3_r * volume);
        Ek /= real_c(atoms.numLocalAtoms);
        currentTemperature << (2_r / 3_r) * Ek;

        ghostLayer.contributeBackGhostToReal(atoms);
        action::VelocityVerlet::postForceIntegrate(atoms, config["dt"].as<real_t>());

        if ((config["enable_output"].as<bool>()) &&
            (step % config["output_interval"].as<int64_t>() == 0))
        {
            util::printTable(step,
                             timer.seconds(),
                             currentTemperature,
                             currentPressure,
                             volume,
                             Ek,
                             LJ.getEnergy() / real_c(atoms.numLocalAtoms),
                             Ek + LJ.getEnergy() / real_c(atoms.numLocalAtoms),
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            io::dumpCSV(std::format("NVT_{:0>6}.csv", step), atoms);
        }
    }
    if (config["enable_output"].as<bool>())
        util::printTableSep(
            "step", "wall time", "T", "p", "V", "E_kin", "E_LJ", "E_total", "Nlocal", "Nghost");
}
}  // namespace mrmd