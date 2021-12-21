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
#include "analysis/AxialDensityProfile.hpp"
#include "analysis/Fluctuation.hpp"
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
    LJ.setCompensationEnergySamplingInterval(
        config["compensation_energy_sampling_interval"].as<idx_t>());
    LJ.setCompensationEnergyUpdateInterval(
        config["compensation_energy_update_interval"].as<idx_t>());
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

    idx_t countTypeA = 0;
    auto policy = Kokkos::RangePolicy<>(0, atoms.numLocalAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t& idx, idx_t& count)
    {
        count += atoms.getType()(idx) == 0 ? 1 : 0;
    };
    Kokkos::parallel_reduce(policy, kernel, countTypeA);
    Kokkos::fence();
    auto rhoA = real_c(countTypeA) / volume;
    auto rhoB = real_c(atoms.numLocalAtoms - countTypeA) / volume;
    //    std::cout << rhoA << " " << rhoB << std::endl;
    action::ThermodynamicForce thermodynamicForce(
        {rhoA, rhoB},
        subdomain,
        config["density_bin_width"].as<real_t>(),
        config["thermodynamic_force_modulation"].as<std::vector<real_t>>());

    std::ofstream fDensityOut1("densityProfile1.txt");
    std::ofstream fDensityOut2("densityProfile2.txt");
    std::ofstream fThermodynamicForceOut1("thermodynamicForce1.txt");
    std::ofstream fThermodynamicForceOut2("thermodynamicForce2.txt");
    std::ofstream fDriftForceCompensation1("driftForce1.txt");
    std::ofstream fDriftForceCompensation2("driftForce2.txt");

    Kokkos::Timer timer;

    if (config["enable_output"].as<bool>())
        util::printTable("step",
                         "wall time",
                         "T",
                         "p",
                         "V",
                         "mu_left",
                         "mu_right",
                         "XrhoA",
                         "XrhoB",
                         "Nlocal",
                         "Nghost");
    if (config["enable_output"].as<bool>())
        util::printTableSep("step",
                            "wall time",
                            "T",
                            "p",
                            "V",
                            "mu_left",
                            "mu_right",
                            "XrhoA",
                            "XrhoB",
                            "Nlocal",
                            "Nghost");
    auto Xrho1 = 0_r;
    auto Xrho2 = 0_r;
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

        if ((step > config["density_start"].as<idx_t>()) &&
            (step % config["density_sampling_interval"].as<idx_t>() == 0))
        {
            thermodynamicForce.sample(atoms);
        }

        if ((step > config["density_start"].as<idx_t>()) &&
            (step % config["density_update_interval"].as<idx_t>() == 0))
        {
            auto densityProfile = analysis::getAxialDensityProfile(atoms.numLocalAtoms,
                                                                   atoms.getPos(),
                                                                   atoms.getType(),
                                                                   2,
                                                                   subdomain.minCorner[0],
                                                                   subdomain.maxCorner[0],
                                                                   100);
            densityProfile.scale(densityProfile.binSize * subdomain.diameter[1] *
                                 subdomain.diameter[2]);
            Xrho1 = analysis::getFluctuation(densityProfile, rhoA, 0);
            Xrho2 = analysis::getFluctuation(densityProfile, rhoA, 1);
            auto h_densityProfile =
                Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), densityProfile.data);
            for (auto i = 0; i < h_densityProfile.extent(0); ++i)
            {
                fDensityOut1 << h_densityProfile(i, 0) << " ";
                fDensityOut2 << h_densityProfile(i, 1) << " ";
            }
            fDensityOut1 << std::endl;
            fDensityOut2 << std::endl;

            thermodynamicForce.update(config["smoothing_sigma"].as<real_t>(),
                                      config["smoothing_intensity"].as<real_t>());
        }

        if (step > config["density_start"].as<idx_t>())
        {
            thermodynamicForce.apply(atoms, weightingFunction);
        }

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
            auto Fth1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                            thermodynamicForce.getForce(0));
            auto Fth2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                            thermodynamicForce.getForce(1));
            auto muLeft = 0_r;
            for (auto i = 0; i < Fth2.extent(0) / 2; ++i)
            {
                muLeft += Fth2(i);
            }
            muLeft *= thermodynamicForce.getForce().binSize;

            auto muRight = 0_r;
            for (auto i = Fth2.extent(0) / 2; i < Fth2.extent(0); ++i)
            {
                muRight += Fth2(i);
            }
            muRight *= thermodynamicForce.getForce().binSize;

            util::printTable(step,
                             timer.seconds(),
                             currentTemperature,
                             currentPressure,
                             volume,
                             muLeft,
                             muRight,
                             Xrho1,
                             Xrho2,
                             atoms.numLocalAtoms,
                             atoms.numGhostAtoms);

            for (auto i = 0; i < Fth1.extent(0); ++i)
            {
                fThermodynamicForceOut1 << Fth1(i) << " ";
            }
            fThermodynamicForceOut1 << std::endl;

            for (auto i = 0; i < Fth2.extent(0); ++i)
            {
                fThermodynamicForceOut2 << Fth2(i) << " ";
            }
            fThermodynamicForceOut2 << std::endl;

            auto h_meanCompensationEnergy = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), LJ.getMeanCompensationEnergy().data);
            for (auto i = 0; i < h_meanCompensationEnergy.extent(0); ++i)
            {
                fDriftForceCompensation1 << h_meanCompensationEnergy(i, 0) << " ";
                fDriftForceCompensation2 << h_meanCompensationEnergy(i, 1) << " ";
            }
            fDriftForceCompensation1 << std::endl;
            fDriftForceCompensation2 << std::endl;

            // io::dumpCSV(fmt::format("spartian_{:0>6}.csv", step), atoms, false);
        }
    }
    if (config["enable_output"].as<bool>())
        util::printTableSep("step",
                            "wall time",
                            "T",
                            "p",
                            "V",
                            "mu_left",
                            "mu_right",
                            "XrhoA",
                            "XrhoB",
                            "Nlocal",
                            "Nghost");

    fDensityOut1.close();
    fDensityOut2.close();
    fThermodynamicForceOut1.close();
    fThermodynamicForceOut2.close();
    fDriftForceCompensation1.close();
    fDriftForceCompensation2.close();
}
}  // namespace mrmd
