#pragma once

#include "data/Particles.hpp"
#include "data/Subdomain.hpp"
#include "datatypes.hpp"

namespace mrmd
{
namespace analysis
{
class MeanSquareDisplacement
{
private:
    VectorView initialPosition_;
    idx_t numParticles_;
    data::Subdomain subdomain_;

public:
    void reset(data::Particles& atoms);
    real_t calc(data::Particles& atoms);

    MeanSquareDisplacement(const data::Subdomain& subdomain)
        : initialPosition_("MeanSquareDisplacement::initialPosition", 0), subdomain_(subdomain)
    {
    }
};

}  // namespace analysis
}  // namespace mrmd