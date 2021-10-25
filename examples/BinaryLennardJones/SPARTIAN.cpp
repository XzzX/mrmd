#include "SPARTIAN.hpp"

#include <fmt/format.h>

#include <fstream>

#include "action/BerendsenBarostat.hpp"
#include "action/BerendsenThermostat.hpp"
#include "action/ContributeMoleculeForceToAtoms.hpp"
#include "action/LJ_IdealGas.hpp"
#include "action/ThermodynamicForce.hpp"
#include "action/UpdateMolecules.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "communication/MultiResGhostLayer.hpp"
#include "io/DumpCSV.hpp"
#include "util/ExponentialMovingAverage.hpp"
#include "util/PrintTable.hpp"
#include "weighting_function/Slab.hpp"

namespace mrmd
{
void spartian(YAML::Node& config,
              data::Molecules& molecules,
              data::Atoms& atoms,
              data::Subdomain& subdomain)
{
    constexpr int64_t estimatedMaxNeighbors = 60;
    constexpr real_t cellRatio = 0.5_r;
    const real_t skin = config["LJ"]["skin"].as<real_t>();
    auto rcVec = config["LJ"]["cutoff"].as<std::vector<real_t>>();
    const real_t rc = *std::max_element(rcVec.begin(), rcVec.end());
    const real_t neighborCutoff = rc + skin;
    auto volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];

    communication::MultiResGhostLayer ghostLayer;
    auto LJ = action::LJ_IdealGas(config["LJ"]["capping"].as<std::vector<real_t>>(),
                                  config["LJ"]["cutoff"].as<std::vector<real_t>>(),
                                  config["LJ"]["sigma"].as<std::vector<real_t>>(),
                                  config["LJ"]["epsilon"].as<std::vector<real_t>>(),
                                  2,
                                  true);
    VerletList verletList;

    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    util::ExponentialMovingAverage currentPressure(
        config["pressure_averaging_coefficient"].as<real_t>());
    util::ExponentialMovingAverage currentTemperature(
        config["temperature_averaging_coefficient"].as<real_t>());
    currentTemperature << analysis::getMeanKineticEnergy(atoms) * 2_r / 3_r;

    auto weightingFunction =
        weighting_function::Slab({config["center"][0].as<real_t>(),
                                  config["center"][1].as<real_t>(),
                                  config["center"][2].as<real_t>()},
                                 config["atomistic_region_diameter"].as<real_t>(),
                                 config["hybrid_region_diameter"].as<real_t>(),
                                 config["lambda_exponent"].as<int64_t>());
    auto rho = real_c(atoms.numLocalAtoms) / volume;
    action::ThermodynamicForce thermodynamicForce(
        rho, subdomain, config["thermodynamic_force_modulation"].as<real_t>());

    std::ofstream fDensityOut("densityProfile.txt");
    std::ofstream fThermodynamicForceOut("thermodynamicForce.txt");
    std::ofstream fDriftForceCompensation("driftForce.txt");

    if (config["enable_output"].as<bool>())
        util::printTable("step", "T", "p", "V", "mu", "Nlocal", "Nghost");
    if (config["enable_output"].as<bool>())
        util::printTableSep("step", "T", "p", "V", "mu", "Nlocal", "Nghost");
    for (auto step = 0; step < config["time_steps"].as<int64_t>(); ++step)
    {
        assert(atoms.numLocalAtoms == molecules.numLocalMolecules);
        assert(atoms.numGhostAtoms == molecules.numGhostMolecules);

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

        // update molecule positions
        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        if (maxAtomDisplacement >= skin * 0.5_r)
        {
            // reset displacement
            maxAtomDisplacement = 0_r;

            ghostLayer.exchangeRealAtoms(molecules, atoms, subdomain);

            //            real_t gridDelta[3] = {neighborCutoff, neighborCutoff, neighborCutoff};
            //            LinkedCellList linkedCellList(atoms.getPos(),
            //                                          0,
            //                                          atoms.numLocalAtoms,
            //                                          gridDelta,
            //                                          subdomain.minCorner.data(),
            //                                          subdomain.maxCorner.data());
            //            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(molecules, atoms, subdomain);
            verletList.build(molecules.getPos(),
                             0,
                             molecules.numLocalMolecules,
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

        action::UpdateMolecules::update(molecules, atoms, weightingFunction);

        auto atomsForce = atoms.getForce();
        Cabana::deep_copy(atomsForce, 0_r);
        auto moleculesForce = molecules.getForce();
        Cabana::deep_copy(moleculesForce, 0_r);

        if (step % config["density_sampling_interval"].as<idx_t>() == 0)
        {
            thermodynamicForce.sample(atoms);
        }

        if (step % config["density_update_interval"].as<idx_t>() == 0)
        {
            thermodynamicForce.update();
        }

        thermodynamicForce.apply(atoms);

        LJ.run(molecules, verletList, atoms);
        action::ContributeMoleculeForceToAtoms::update(molecules, atoms);

        if (step < 201)
        {
            currentPressure = util::ExponentialMovingAverage(
                config["pressure_averaging_coefficient"].as<real_t>());
            currentTemperature = util::ExponentialMovingAverage(
                config["temperature_averaging_coefficient"].as<real_t>());
        }
        auto Ek = analysis::getKineticEnergy(atoms);
        //        currentPressure << 2_r * (Ek - LJ.getVirial()) / (3_r * volume);
        Ek /= real_c(atoms.numLocalAtoms);
        currentTemperature << (2_r / 3_r) * Ek;

        ghostLayer.contributeBackGhostToReal(atoms);
        action::VelocityVerlet::postForceIntegrate(atoms, config["dt"].as<real_t>());

        if ((config["enable_output"].as<bool>()) &&
            (step % config["output_interval"].as<int64_t>() == 0))
        {
            // calc chemical potential
            auto Fth = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                           thermodynamicForce.getForce().data);
            auto mu = 0_r;
            for (auto i = 0; i < Fth.extent(0) / 2; ++i)
            {
                mu += Fth(i);
            }
            mu *= thermodynamicForce.getForce().binSize;

            util::printTable(step,
                             currentTemperature,
                             currentPressure,
                             volume,
                             mu,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            fThermodynamicForceOut << thermodynamicForce.getForce() << std::endl;
            fDriftForceCompensation << LJ.getMeanCompensationEnergy() << std::endl;

            io::dumpCSV(fmt::format("spartian_{:0>6}.csv", step), atoms, false);
        }
    }
    if (config["enable_output"].as<bool>())
        util::printTableSep("step", "T", "p", "V", "mu", "Nlocal", "Nghost");

    fDensityOut.close();
    fThermodynamicForceOut.close();
    fDriftForceCompensation.close();
}
}  // namespace mrmd