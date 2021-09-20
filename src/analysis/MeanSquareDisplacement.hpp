#pragma once

#include "data/Particles.hpp"
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

public:
    void reset(data::Particles& atoms);
    real_t calc(data::Particles& atoms);

    MeanSquareDisplacement() : initialPosition_("MeanSquareDisplacement::initialPosition", 0) {}
};

}  // namespace analysis
}  // namespace mrmd