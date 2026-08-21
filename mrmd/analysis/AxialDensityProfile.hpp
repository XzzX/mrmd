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

#include <vector>

#include "assert/assert.hpp"
#include "data/Atoms.hpp"
#include "data/MultiHistogram.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{ /**
   * Calculate a discretized particle number profile along an axis.
   * Out-of-bounds values are discarded.
   */
data::MultiHistogram getAxialParticleNumberProfile(const data::Atoms& atoms,
                                                   const real_t min,
                                                   const real_t max,
                                                   const idx_t numBins,
                                                   const AXIS axis);

class PlaneWiseMassDensityProfile
{
    // see eq. (4) in https://doi.org/10.1063/1.471718
private:
    data::MultiHistogram averageMassDensityProfile_;
    idx_t numberOfSamples_ = 0;
    data::Atoms atomsBeforeMeasurement_;
    const AXIS axis_;

public:
    PlaneWiseMassDensityProfile(const data::Atoms& atoms,
                                const data::Subdomain& subdomain,
                                const real_t binWidth,
                                const AXIS axis)
        : averageMassDensityProfile_("plane-wise-mass-density-profile",
                                     subdomain.minCorner[to_underlying(axis)],
                                     subdomain.maxCorner[to_underlying(axis)],
                                     idx_c(std::ceil((subdomain.maxCorner[to_underlying(axis)] -
                                                      subdomain.minCorner[to_underlying(axis)]) /
                                                     binWidth)),
                                     atoms.getNumTypes()),
          atomsBeforeMeasurement_(atoms),
          axis_(axis)
    {
        MRMD_HOST_CHECK_FLOAT_EQUAL(
            averageMassDensityProfile_.binSize, binWidth, "requested bin size is not achieved");

        MRMD_HOST_CHECK_GREATER(atoms.getNumTypes(), 0);
    }

    void startMeasuringCrossingParticles(const data::Atoms& atoms)
    {
        data::deep_copy(atomsBeforeMeasurement_, atoms);
    };

    inline auto getAverageProfile() const { return averageMassDensityProfile_; }
    inline auto getAverageProfile(const idx_t& typeId) const
    {
        assert(typeId < averageMassDensityProfile_.data.extent(1));
        assert(typeId >= 0);
        return Kokkos::subview(averageMassDensityProfile_.data, Kokkos::ALL(), typeId);
    }
    
    void stopMeasuringCrossingParticles(const data::Atoms& atoms,
                                        const data::Subdomain subdomain,
                                        const real_t dt)
    {
        MRMD_HOST_CHECK_EQUAL(
            atomsBeforeMeasurement_.size(),
            atoms.size(),
            "The number of particles is not allowed to change between "
            "startMeasuringCrossingParticles and stopMeasuringCrossingParticles.");
        auto numAtoms = atoms.numLocalAtoms + atoms.numGhostAtoms;
        auto numTypes = atoms.getNumTypes();
        auto positionsBefore = atomsBeforeMeasurement_.getPos();
        auto positionsAfter = atoms.getPos();
        auto velocities = atoms.getVel();
        auto types = atoms.getType();
        auto masses = atoms.getMass();

        data::MultiHistogram instantaneousMassDensityProfile("instantaneous-mass-density-profile",
                                                             averageMassDensityProfile_.min,
                                                             averageMassDensityProfile_.max,
                                                             averageMassDensityProfile_.numBins,
                                                             numTypes);
        MultiScatterView scatter(instantaneousMassDensityProfile.data);

        auto policy = Kokkos::RangePolicy<>(0, numAtoms);
        auto kernel = KOKKOS_LAMBDA(const idx_t idx)
        {
            MRMD_DEVICE_ASSERT_GREATEREQUAL(types(idx), 0);
            MRMD_DEVICE_ASSERT_LESS(types(idx), numTypes);
            auto posBefore = positionsBefore(idx, to_underlying(axis_));
            auto posAfter = positionsAfter(idx, to_underlying(axis_));
            auto lower = posBefore < posAfter ? posBefore : posAfter;
            auto upper = posBefore < posAfter ? posAfter : posBefore;

            auto firstBin = idx_c(std::floor((lower - instantaneousMassDensityProfile.min) *
                                             instantaneousMassDensityProfile.inverseBinSize));
            auto lastBin = idx_c(std::floor((upper - instantaneousMassDensityProfile.min) *
                                            instantaneousMassDensityProfile.inverseBinSize));
            if (firstBin < 0) firstBin = 0;
            if (lastBin >= instantaneousMassDensityProfile.numBins)
                lastBin = instantaneousMassDensityProfile.numBins - 1;

            auto access = scatter.access();
            for (auto bin = firstBin; bin <= lastBin; ++bin)
            {
                access(bin, types(idx)) +=
                    masses(idx) / Kokkos::abs(velocities(idx, to_underlying(axis_)));
            }
        };
        Kokkos::parallel_for(policy, kernel);
        Kokkos::Experimental::contribute(instantaneousMassDensityProfile.data, scatter);
        Kokkos::fence();

        instantaneousMassDensityProfile.scale(1_r / (dt * subdomain.getAreaNormalToAxis(axis_)));

        data::cumulativeMovingAverage(
            averageMassDensityProfile_, instantaneousMassDensityProfile, real_c(numberOfSamples_));
        ++numberOfSamples_;
    };

    void reset()
    {
        MRMD_HOST_CHECK_GREATER(
            numberOfSamples_,
            0,
            "Cannot reset AxialAverageProfile because no samples have been taken yet.");

        Kokkos::deep_copy(averageMassDensityProfile_.data, 0_r);
        numberOfSamples_ = 0;
    }
};
}  // namespace analysis
}  // namespace mrmd