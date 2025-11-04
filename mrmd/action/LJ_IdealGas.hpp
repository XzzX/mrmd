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

#include "LennardJones.hpp"
#include "assert/assert.hpp"
#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
#include "data/MultiHistogram.hpp"
#include "datatypes.hpp"
#include "weighting_function/CheckRegion.hpp"

namespace mrmd::action
{
void updateMeanCompensationEnergy(data::MultiHistogram& compensationEnergy,
                                  data::MultiHistogram& compensationEnergyCounter,
                                  data::MultiHistogram& meanCompensationEnergy,
                                  const real_t runningAverageFactor = 10_r);

class LJ_IdealGas
{
private:
    idx_t numTypes_;
    
    impl::CappedLennardJonesPotential LJ_;
    real_t rcSqr_ = 0_r;

    data::Molecules::pos_t moleculesPos_;
    data::Molecules::force_t::atomic_access_slice moleculesForce_;
    data::Molecules::lambda_t moleculesLambda_;
    data::Molecules::modulated_lambda_t moleculesModulatedLambda_;
    data::Molecules::grad_lambda_t moleculesGradLambda_;
    data::Molecules::atoms_offset_t moleculesAtomsOffset_;
    data::Molecules::num_atoms_t moleculesNumAtoms_;

    data::Atoms::pos_t atomsPos_;
    data::Atoms::force_t::atomic_access_slice atomsForce_;
    data::Atoms::type_t atomsType_;

    static constexpr idx_t COMPENSATION_ENERGY_BINS = 200;

    data::MultiHistogram compensationEnergy_;
    MultiScatterView compensationEnergyScatter_;

    data::MultiHistogram compensationEnergyCounter_;

    data::MultiHistogram meanCompensationEnergy_;

    bool isDriftCompensationSamplingRun_ = false;

    HalfVerletList verletList_;

    idx_t runCounter_ = 0;

    idx_t compensationEnergySamplingInterval = 200;
    idx_t compensationEnergyUpdateInveral = 20000;

public:
    void setCompensationEnergySamplingInterval(const idx_t& interval)
    {
        compensationEnergySamplingInterval = interval;
    }
    void setCompensationEnergyUpdateInterval(const idx_t& interval)
    {
        compensationEnergyUpdateInveral = interval;
    }
    const auto& getMeanCompensationEnergy() const { return meanCompensationEnergy_; }

    /**
     * Loop over molecules
     *
     * @param alpha first molecule index
     */
    KOKKOS_FUNCTION void operator()(const idx_t& alpha, real_t& sumEnergy) const;

    real_t run(data::Molecules& molecules, HalfVerletList& verletList, data::Atoms& atoms);

    LJ_IdealGas(const real_t& cappingDistance,
                const real_t& rc,
                const real_t& sigma,
                const real_t& epsilon,
                const bool doShift);

    LJ_IdealGas(const std::vector<real_t>& cappingDistance,
                const std::vector<real_t>& rc,
                const std::vector<real_t>& sigma,
                const std::vector<real_t>& epsilon,
                const idx_t numTypes,
                const bool doShift);
};

}  // namespace mrmd::action