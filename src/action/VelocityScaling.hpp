#pragma once

#include "data/Atoms.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace action
{
/**
 * velocity rescaling thermostat
 * DOI: 10.1007/978-3-540-68095-6
 * EQ: 3.43
 */
class VelocityScaling
{
private:
    real_t targetTemperature_;
    real_t gamma_;

public:
    /**
     * @param degreesOfFreedom in 3D typically 3N - 3
     */
    void apply(data::Atoms& atoms, const real_t& degreesOfFreedom) const;

    void set(const real_t gamma, const real_t targetTemperature)
    {
        assert(gamma >= 0_r);
        assert(gamma <= 1_r);
        gamma_ = gamma;
        targetTemperature_ = targetTemperature;
    }

    VelocityScaling(const real_t gamma, const real_t targetTemperature)
    {
        set(gamma, targetTemperature);
    }
};
}  // namespace action
}  // namespace mrmd