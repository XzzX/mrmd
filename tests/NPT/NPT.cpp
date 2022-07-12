#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "Cabana_NeighborList.hpp"
#include "action/BerendsenBarostat.hpp"
#include "action/BerendsenThermostat.hpp"
#include "action/LangevinThermostat.hpp"
#include "action/LennardJones.hpp"
#include "action/VelocityVerlet.hpp"
#include "analysis/KineticEnergy.hpp"
#include "analysis/Pressure.hpp"
#include "analysis/SystemMomentum.hpp"
#include "communication/GhostLayer.hpp"
#include "data/Atoms.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"
#include "io/DumpCSV.hpp"
#include "io/RestoreTXT.hpp"
#include "util/EnvironmentVariables.hpp"
#include "util/ExponentialMovingAverage.hpp"
#include "util/PrintTable.hpp"

using namespace mrmd;

struct Config
{
    static constexpr bool bOutput = true;
    static constexpr idx_t outputInterval = -1;

    static constexpr idx_t nsteps = 2001;
    static constexpr real_t rc = 2.5;
    static constexpr real_t skin = 0.3;
    static constexpr real_t neighborCutoff = rc + skin;
    static constexpr real_t sigma = real_t(1);
    static constexpr real_t epsilon = real_t(1);
    static constexpr real_t dt = 0.005;
    static constexpr real_t gamma = real_t(0.1);

    static constexpr real_t Lx = real_t(16.92926877476863);
    static constexpr real_t rho = real_t(0.8442);

    static constexpr real_t cell_ratio = real_t(1.0);

    static constexpr idx_t estimatedMaxNeighbors = 60;

    static constexpr real_t weightingFactor = real_t(0.02);
};

data::Atoms fillDomainWithAtomsSC(const data::Subdomain &subdomain,
                                  const idx_t &numAtoms,
                                  const real_t &maxVelocity)
{
    auto RNG = Kokkos::Random_XorShift1024_Pool<>(1234);

    data::Atoms atoms(numAtoms);

    auto pos = atoms.getPos();
    auto vel = atoms.getVel();
    auto mass = atoms.getMass();

    auto policy = Kokkos::RangePolicy<>(0, numAtoms);
    auto kernel = KOKKOS_LAMBDA(const idx_t idx)
    {
        auto randGen = RNG.get_state();
        pos(idx, 0) = randGen.drand() * subdomain.diameter[0] + subdomain.minCorner[0];
        pos(idx, 1) = randGen.drand() * subdomain.diameter[1] + subdomain.minCorner[1];
        pos(idx, 2) = randGen.drand() * subdomain.diameter[2] + subdomain.minCorner[2];

        vel(idx, 0) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        vel(idx, 1) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        vel(idx, 2) = (randGen.drand() - real_t(0.5)) * maxVelocity;
        RNG.free_state(randGen);

        mass(idx) = real_t(1);
    };
    Kokkos::parallel_for("fillDomainWithAtomsSC", policy, kernel);

    atoms.numLocalAtoms = numAtoms;
    atoms.numGhostAtoms = 0;
    return atoms;
}

struct Input
{
    real_t targetTemperature;
    real_t targetPressure;
};

std::ostream &operator<<(std::ostream &os, const Input &input)
{
    os << "T: " << input.targetTemperature << " | "
       << "p: " << input.targetPressure;
    return os;
}

class NPT : public ::testing::TestWithParam<Input>
{
protected:
    // void SetUp() override {}
    // void TearDown() override {}

    data::Subdomain subdomain;
    real_t volume;
    data::Atoms atoms = data::Atoms(0);
    real_t rho;
    communication::GhostLayer ghostLayer;
    action::LennardJones LJ = action::LennardJones(real_t(0), real_t(0), real_t(0), real_t(0));

    HalfVerletList verletList;

public:
    NPT()
        : subdomain({real_t(0), real_t(0), real_t(0)},
                    {Config::Lx, Config::Lx, Config::Lx},
                    Config::neighborCutoff),
          volume(subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2]),
          atoms(fillDomainWithAtomsSC(subdomain, idx_c(Config::rho * volume), real_t(1))),
          rho(real_c(atoms.numLocalAtoms) / volume),
          LJ(Config::rc, Config::sigma, Config::epsilon, real_t(0.7) * Config::sigma)
    {
    }
};

TEST_P(NPT, pressure)
{
    auto targetTemperature = GetParam().targetTemperature;
    auto targetPressure = GetParam().targetPressure;

    real_t maxAtomDisplacement = std::numeric_limits<real_t>::max();
    util::ExponentialMovingAverage p(Config::weightingFactor);
    util::ExponentialMovingAverage T(Config::weightingFactor);
    T << analysis::getMeanKineticEnergy(atoms) * real_t(2) / real_t(3);
    for (auto step = 0; step < Config::nsteps; ++step)
    {
        maxAtomDisplacement += action::VelocityVerlet::preForceIntegrate(atoms, Config::dt);

        if ((step > 200) && (step % 100 == 0))
        {
            action::BerendsenBarostat::apply(
                atoms, p, targetPressure, Config::gamma * real_t(0.1), subdomain);
            volume = subdomain.diameter[0] * subdomain.diameter[1] * subdomain.diameter[2];
            maxAtomDisplacement = std::numeric_limits<real_t>::max();
        }
        action::BerendsenThermostat::apply(atoms, T, targetTemperature, Config::gamma);

        if (maxAtomDisplacement >= Config::skin * real_t(0.5))
        {
            // reset displacement
            maxAtomDisplacement = real_t(0);

            ghostLayer.exchangeRealAtoms(atoms, subdomain);

            real_t gridDelta[3] = {
                Config::neighborCutoff, Config::neighborCutoff, Config::neighborCutoff};
            LinkedCellList linkedCellList(atoms.getPos(),
                                          0,
                                          atoms.numLocalAtoms,
                                          gridDelta,
                                          subdomain.minCorner.data(),
                                          subdomain.maxCorner.data());
            atoms.permute(linkedCellList);

            ghostLayer.createGhostAtoms(atoms, subdomain);
            verletList.build(atoms.getPos(),
                             0,
                             atoms.numLocalAtoms,
                             Config::neighborCutoff,
                             Config::cell_ratio,
                             subdomain.minGhostCorner.data(),
                             subdomain.maxGhostCorner.data(),
                             Config::estimatedMaxNeighbors);
        }
        else
        {
            ghostLayer.updateGhostAtoms(atoms, subdomain);
        }

        auto force = atoms.getForce();
        Cabana::deep_copy(force, real_t(0));

        LJ.apply(atoms, verletList);

        if (step < 201)
        {
            p = util::ExponentialMovingAverage(Config::weightingFactor);
            T = util::ExponentialMovingAverage(Config::weightingFactor);
        }
        auto Ek = analysis::getKineticEnergy(atoms);
        p << real_t(2) * (Ek - LJ.getVirial()) / (real_t(3) * volume);
        Ek /= real_c(atoms.numLocalAtoms);
        T << (real_t(2) / real_t(3)) * Ek;

        ghostLayer.contributeBackGhostToReal(atoms);
        action::VelocityVerlet::postForceIntegrate(atoms, Config::dt);

        if (Config::bOutput && (step % Config::outputInterval == 0))
        {
            //            std::cout << step << " " << T << " " << p << " " << volume << std::endl;
        }
    }

    EXPECT_NEAR(T, targetTemperature, real_t(0.1));
    EXPECT_NEAR(p, targetPressure, real_t(0.2));
}

INSTANTIATE_TEST_SUITE_P(Pressure,
                         NPT,
                         ::testing::Values(Input{real_t(2.8), real_t(9.1)},
                                           Input{real_t(2.5), real_t(8.5)},
                                           Input{real_t(2.0), real_t(8.0)}));

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::ScopeGuard scope_guard(argc, argv);
    return RUN_ALL_TESTS();
}
