#pragma once

#include "data/Atoms.hpp"
#include "data/Molecules.hpp"
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
    idx_t numItems_;

public:
    /**
     * Store current positions as a starting point.
     */
    void reset(data::Atoms &atoms);

    void reset(data::Molecules &molecules);

    /**
     * Compare current positions against the starting point
     * @return mean squre displacement
     */
    real_t calc(data::Atoms &atoms, const data::Subdomain &subdomain);

    real_t calc(data::Molecules &molecules, const data::Subdomain &subdomain);

    MeanSquareDisplacement() : initialPosition_("MeanSquareDisplacement::initialPosition", 0) {}
};

}  // namespace analysis
}  // namespace mrmd